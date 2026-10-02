"""来源：本项目原创。停止预测、原实例执行、扰动反例与离线性能测量。"""
import argparse
from dataclasses import replace
import gzip
import html
import json
from math import hypot
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.stop_forecast import (StopRecipe, FrozenForecast, PhysicsBranch, forecast_stop,
                                    binding, check_binding, rollout, integration_values, digest, CLOCK)
from drone_nav.realvision import sha, dump
from tools.stopping_policy_experiment import ALL_CASES, read, independent_hold
from tools.braking_measurement_experiment import independent_metrics, canonical
from tools.object_probe import check_manifest

PARENT = ROOT/'work/stopping-policy-01'
OWN = ('drone_nav/stop_forecast.py', 'tools/stop_forecast_experiment.py',
       'tests/test_stop_forecast.py', 'docs/STOP_FORECAST_PROTOCOL.md')
TRIALS = [(c.name, p, None) for c in ALL_CASES for p in ('original', 'attitude')]
TRIALS += [('x20', p, f) for p in ('original', 'attitude') for f in ('pitch_pulse', 'thrust_loss')]


def sources():
    check_manifest(PARENT)
    parent = json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    for name, value in parent['sources'].items():
        if sha(ROOT/name) != value: raise ValueError('frozen source differs: '+name)
    return {**parent['sources'], **{name: sha(ROOT/name) for name in OWN}}


def compare(predicted, actual):
    if len(predicted['states']) != len(actual['states']): raise ValueError('trajectory length differs')
    errors, speed_errors, axis_errors = [], [], [0., 0., 0.]
    outside = []
    envelope = predicted['envelope']
    for i, (p, a) in enumerate(zip(predicted['states'], actual['states'])):
        if p['time_s'] != a['time_s']: raise ValueError('trajectory clock differs')
        delta = [x-y for x, y in zip(a['position'], p['position'])]
        errors.append(hypot(*delta))
        speed_errors.append(hypot(*(x-y for x, y in zip(a['velocity'], p['velocity']))))
        axis_errors = [max(old, abs(new)) for old, new in zip(axis_errors, delta)]
        if any(not envelope['center_lower_m'][j]-.02-1e-9 <= a['position'][j] <= envelope['center_upper_m'][j]+.02+1e-9 for j in range(3)):
            outside.append(i)
    return dict(states_exact=predicted['states'] == actual['states'], commands_exact=predicted['commands'] == actual['commands'],
                max_position_error_m=max(errors), max_velocity_error_mps=max(speed_errors), max_axis_errors_m=axis_errors,
                envelope_outside_samples=len(outside), first_outside_index=outside[0] if outside else None,
                sampled_states=len(errors), position_errors=errors)


def run_trial(index):
    name, policy, fault = TRIALS[index]
    case = next(c for c in ALL_CASES if c.name == name)
    parent_path = PARENT/name/policy/'evidence.json.gz'
    old = read(parent_path)
    prefix = old['commands'][:old['request_index']]
    recipe = StopRecipe(policy, case.control_seconds, case.delay_seconds, direction=case.direction)
    with QuadrotorPhysics() as physics:
        physics.reset(old['initial_position'])
        for command in prefix: physics.step(command['motors'])
        if physics.state() != old['states'][old['request_index']]: raise ValueError('prefix endpoint differs')
        initial = binding(physics)
        frozen, elapsed = forecast_stop(physics, recipe)
        if binding(physics) != initial: raise ValueError('forecast changed actual instance')
        check_binding(frozen, initial, recipe, clock_id=CLOCK)
        negatives = []
        def reject(label, operation):
            try: operation()
            except ValueError: negatives.append(label)
            else: raise ValueError('failed to reject '+label)
        reject('wrong_policy', lambda: check_binding(frozen, initial, replace(recipe, policy='attitude' if policy == 'original' else 'original'), clock_id=CLOCK))
        reject('wrong_clock', lambda: check_binding(frozen, initial, recipe, clock_id='wall-clock'))
        reject('changed_content', lambda: replace(frozen, content=frozen.content+' ').record())
        record = frozen.record()
        with PhysicsBranch(physics) as stale:
            stale.step(record['last_motors'])
            changed = dict(initial, state_sha256=digest(integration_values(stale)))
            reject('advanced_state', lambda: check_binding(frozen, changed, recipe, clock_id=CLOCK))
        # Closing an independent branch must not close or modify its parent.
        try: stale.state()
        except RuntimeError: pass
        else: raise ValueError('closed branch remained usable')
        if binding(physics) != initial: raise ValueError('binding test changed actual instance')
        actual = rollout(physics, recipe, record['last_motors'], fault=fault)
        predicted = record['prediction']
        comparison = compare(predicted, actual)
        if fault is None:
            if not comparison['states_exact'] or not comparison['commands_exact']:
                raise ValueError('nominal forecast does not reproduce execution')
            if actual['states'] != old['states'][old['request_index']:] or [c['motors'] for c in actual['commands']] != [c['motors'] for c in old['commands'][old['request_index']:]]:
                raise ValueError('nominal execution differs from frozen parent')
        elif comparison['states_exact']:
            raise ValueError('injected actuator fault had no observed effect')
        audits = [independent_metrics(r) for r in (predicted, actual)]
        for value in (predicted, actual): independent_hold(value)
        raw = canonical(dict(name=name, policy=policy, fault=fault, parent_input=str(parent_path.relative_to(ROOT)).replace('\\', '/'),
                             parent_input_sha256=sha(parent_path), prefix_steps=len(prefix), forecast_sha256=frozen.sha256,
                             forecast=record, actual=actual, comparison=comparison,
                             binding_rejections=negatives, independent_metrics=audits, independent_hold=True,
                             original_unchanged=True, branch_closed_without_closing_parent=True))
    return raw, elapsed


def motor_replay(raw):
    old = read(ROOT/raw['parent_input'])
    with QuadrotorPhysics() as physics:
        physics.reset(old['initial_position'])
        for command in old['commands'][:raw['prefix_steps']]: physics.step(command['motors'])
        if physics.state() != raw['actual']['states'][0]: raise ValueError('actual replay entry differs')
        for i, command in enumerate(raw['actual']['commands'], 1):
            physics.step(command['motors'])
            if physics.state() != raw['actual']['states'][i]: raise ValueError('actual motor replay differs')


def worker(index, folder, verify):
    raw, elapsed = run_trial(index)
    evidence = folder/'evidence.json.gz'
    if verify:
        if read(evidence) != raw: raise ValueError('forecast and execution rebuild differ')
        motor_replay(raw)
    else:
        with evidence.open('xb') as stream:
            stream.write(gzip.compress(json.dumps(raw, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
        dump(folder/'timing.json', dict(prediction_seconds=elapsed, budget_seconds=.02,
                                        includes='clone, rollout, analysis, digest, free branch',
                                        excludes='model creation, prefix replay, binding consumption, disk write',
                                        simulation_clock_paused=True))
    return dict(index=index, verified=verify, steps=len(raw['actual']['commands']), measured_prediction_seconds=elapsed,
                max_position_error_m=raw['comparison']['max_position_error_m'], outside=raw['comparison']['envelope_outside_samples'])


def make_chart(trial):
    points = trial['points']
    high = max(max(p[1], p[2]) for p in points)
    low = min(0., min(min(p[1], p[2]) for p in points))
    high = max(high, .001)+(high-low)*.15
    y = lambda v: 220-(v-low)/(high-low)*180
    lines = []
    for field, color in ((1, '#126f88'), (2, '#c16b30')):
        value = ' '.join(f'{60+p[0]/8*675:.2f},{y(p[field]):.2f}' for p in points)
        lines.append(f'<polyline points="{value}" stroke="{color}" stroke-width="2.5" fill="none"/>')
    return f'''<svg viewBox="0 0 790 265" role="img" aria-label="预测与实际沿入口方向位移"><path d="M60 40V220H735" fill="none" stroke="#789"/>{''.join(lines)}<text x="3" y="44">{high:.3f}</text><text x="3" y="220">{low:.3f}</text><text x="65" y="24">位移 m · 蓝：预测，橙：实际；重合时蓝线被覆盖</text><text x="60" y="250">0 秒</text><text x="700" y="250">8 秒</text></svg>'''


def page(report):
    options, panels, rows = [], [], []
    for i, trial in enumerate(report['trials']):
        label = trial['label']
        a = trial['comparison']
        options.append(f'<option value="{i}">{label}</option>')
        panels.append(f'''<section class="trial" data-trial="{i}" style="display:{'block' if i==0 else 'none'}"><h2>{label}</h2>
        <p>预测与实际状态完全相同：{'是' if a['states_exact'] else '否'}。最大三维位置差 <b>{a['max_position_error_m']:.6f} m</b>；速度差 {a['max_velocity_error_mps']:.6f} m/s。</p>
        <p>越出预测中心包围盒加 0.02 米的采样数：<b>{a['envelope_outside_samples']}</b>（不是事故数）。预测总计算 <b>{trial['prediction_seconds']*1000:.1f} ms</b>；20 ms 周期预算：{'满足本次测量' if trial['prediction_seconds'] <= .02 else '超出'}。</p>
        {make_chart(trial)}<p>四类错误绑定均拒绝；摘要匹配仍不授权执行。实测耗时不含模型创建、前段准备、结果消费和写盘；预测期间原仿真停表。</p>
        <details><summary>查看空间包围、保持窗口与误差</summary><pre>{html.escape(json.dumps(dict(envelope=trial['envelope'], predicted_stop=trial['predicted_stop'], actual_stop=trial['actual_stop'], comparison=a), ensure_ascii=False, indent=2))}</pre></details></section>''')
        rows.append(f'<tr><td>{label}</td><td>{a["max_position_error_m"]:.6f}</td><td>{a["envelope_outside_samples"]}</td><td>{trial["prediction_seconds"]*1000:.1f}</td></tr>')
    s = report['summary']
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>停止轨迹预测与执行</title><style>body{margin:0;padding:24px;background:#edf2f5;color:#193346;font:16px/1.7 system-ui}main{max-width:1080px;margin:auto}h1{font-size:30px}section{background:#fff;border-radius:12px;padding:22px;margin:20px 0}select{font:inherit;padding:9px;max-width:100%}svg{width:100%}svg text{font-size:13px;fill:#586977}pre{overflow:auto;max-height:350px;font-size:13px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #dde5e8;white-space:nowrap}.note{color:#89501e}a{color:#126f88}</style><main><h1>先预测停止轨迹，再检查执行偏差</h1><section><p>30 条同模型对照 + 4 条执行扰动 · 独立数据副本 · 完整积分状态绑定</p><p class="note">同模型预测一致不等于现实预测准确。空间包围仅覆盖有限离散样本；没有视觉空闲证明、连续时间误差界或导航授权。无人机不会像本轮原仿真那样等待预测时停表。</p><p>本次预测耗时中位数 '''+f'{s["median_prediction_seconds"]*1000:.1f} ms，最大 {s["max_prediction_seconds"]*1000:.1f} ms。'+'''</p><label for="trial">选择条件：</label><select id="trial">'''+''.join(options)+'''</select></section>'''+''.join(panels)+'''<section><h2>全部结果</h2><div class="scroll"><table><tr><th>案例 / 策略 / 扰动</th><th>最大位置差 m</th><th>越界采样</th><th>预测耗时 ms</th></tr>'''+''.join(rows)+'''</table></div></section><section><h2>复核资料</h2><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定实验协议</a></p><p>每条保存预测、实际推力和完整状态。重放另行测时，原测速记录保留；页面显示正式运行的数据。</p></section></main><script>document.getElementById('trial').addEventListener('change',function(){document.querySelectorAll('.trial').forEach(e=>e.style.display=e.dataset.trial===this.value?'block':'none')})</script></html>'''


def run(output, verify):
    fixed = sources()
    if verify:
        count = check_manifest(output)
        saved = json.loads((output/'report.json').read_text(encoding='utf-8'))
        if saved['sources'] != fixed: raise ValueError('source hashes differ')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'protocol.md').write_bytes((ROOT/'docs/STOP_FORECAST_PROTOCOL.md').read_bytes())
    trials = []
    for index, (name, policy, fault) in enumerate(TRIALS):
        folder = output/f'{index:02d}-{name}-{policy}-{fault or "nominal"}'
        if not verify: folder.mkdir()
        command = [sys.executable, '-X', 'faulthandler', str(Path(__file__).resolve()), '--worker', str(index), '--output', str(folder)]
        if verify: command.append('--verify')
        result = subprocess.run(command, capture_output=True, encoding='utf-8', errors='replace', timeout=60)
        if not verify:
            (folder/'stdout.log').write_text(result.stdout, encoding='utf-8')
            (folder/'stderr.log').write_text(result.stderr, encoding='utf-8')
        if result.returncode:
            print(result.stdout, result.stderr, flush=True)
            raise RuntimeError(f'worker {index} failed with {result.returncode}; no retry')
        fresh = json.loads(result.stdout)
        raw = read(folder/'evidence.json.gz')
        timing = json.loads((folder/'timing.json').read_text(encoding='utf-8'))
        predicted, actual = raw['forecast']['prediction'], raw['actual']
        entry, axis = actual['states'][0], actual['metrics']['evaluation_direction']
        points = [[a['time_s']-entry['time_s'],
                   sum((p['position'][j]-entry['position'][j])*axis[j] for j in range(2)),
                   sum((a['position'][j]-entry['position'][j])*axis[j] for j in range(2))]
                  for p, a in zip(predicted['states'][::10], actual['states'][::10])]
        comparison = {k:v for k,v in raw['comparison'].items() if k != 'position_errors'}
        trials.append(dict(label=f'{name} / {policy} / {fault or "nominal"}', name=name, policy=policy, fault=fault,
                           input=f'{folder.name}/evidence.json.gz', prediction_seconds=timing['prediction_seconds'],
                           prefix_steps=raw['prefix_steps'], actual_steps=len(actual['commands']), prediction_steps=len(predicted['commands']),
                           comparison=comparison, points=points, envelope=predicted['envelope'],
                           predicted_stop=predicted['stop'], actual_stop=actual['stop'], binding_rejections=raw['binding_rejections']))
        print(index, name, policy, fault or 'nominal', 'verified' if verify else 'recorded',
              'error', round(comparison['max_position_error_m'], 6), 'outside', comparison['envelope_outside_samples'],
              'fresh_ms', round(fresh['measured_prediction_seconds']*1000, 1), flush=True)
    if fixed != sources(): raise ValueError('sources changed')
    durations = [t['prediction_seconds'] for t in trials]
    summary = dict(trials=len(trials), nominal_exact=sum(t['fault'] is None and t['comparison']['states_exact'] for t in trials),
                   disturbed_trials=sum(t['fault'] is not None for t in trials),
                   disturbed_outside=sum(t['fault'] is not None and t['comparison']['envelope_outside_samples'] > 0 for t in trials),
                   source_prefix_steps=sum(t['prefix_steps'] for t in trials), actual_stop_steps=sum(t['actual_steps'] for t in trials),
                   prediction_steps=sum(t['prediction_steps'] for t in trials),
                   median_prediction_seconds=statistics.median(durations), max_prediction_seconds=max(durations),
                   within_20ms=sum(t <= .02 for t in durations), within_500ms=sum(t <= .5 for t in durations),
                   binding_rejections=sum(len(t['binding_rejections']) for t in trials))
    report = dict(sources=fixed, parent_sha256=sha(PARENT/'report.json'), trials=trials, summary=summary,
                  new_model_calls=0, real_flights=0, flight_authorized=False)
    if verify:
        if report != saved or page(report) != (output/'demo.html').read_text(encoding='utf-8'): raise ValueError('report/page reconstruction differs')
        if (output/'protocol.md').read_bytes() != (ROOT/'docs/STOP_FORECAST_PROTOCOL.md').read_bytes() or check_manifest(output) != count: raise ValueError('archive/protocol changed')
    else:
        dump(output/'report.json', report)
        (output/'demo.html').write_text(page(report), encoding='utf-8')
        dump(output/'manifest.json', {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()})
        count = check_manifest(output)
    return dict(verified=verify, files=count, sources=len(fixed), **summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--worker', type=int, choices=range(len(TRIALS)))
    args = parser.parse_args()
    value = worker(args.worker, args.output.resolve(), args.verify) if args.worker is not None else run(args.output.resolve(), args.verify)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
