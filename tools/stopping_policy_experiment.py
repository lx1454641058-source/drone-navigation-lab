"""来源：本项目原创。固定新旧停止策略对照，保留每个物理状态和实际推力。"""
import argparse
from dataclasses import asdict
import gzip
import html
import json
from math import hypot
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drone_nav.braking_measurement import CASES, BrakeCase, DT, analyze, stable_window
from drone_nav.flight_control import control
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.stopping_policy import AttitudeStop
from drone_nav.realvision import sha, dump
from tools.braking_measurement_experiment import independent_metrics, canonical
from tools.object_probe import check_manifest

PARENT = ROOT/'work/braking-measurement-01'
OWN = ('drone_nav/stopping_policy.py', 'tools/stopping_policy_experiment.py',
       'tests/test_stopping_policy.py', 'docs/STOPPING_POLICY_PROTOCOL.md')
EXTRA = (BrakeCase('new07', .07, ramp_seconds=4.),
         BrakeCase('new15diagonal', .15, direction=(.6, -.8), ramp_seconds=4.),
         BrakeCase('newearly08', .2, ramp_seconds=.8),
         BrakeCase('newdelay10', .15, ramp_seconds=3., delay_seconds=.1),
         BrakeCase('newperiod10', .15, direction=(-.8, .6), ramp_seconds=4., control_seconds=.01),
         BrakeCase('newpitchpulse', .15, ramp_seconds=3.))
ALL_CASES = CASES+EXTRA


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def source_hashes():
    check_manifest(PARENT)
    report = json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    for path, digest in report['sources'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('frozen source changed: '+path)
    return {**report['sources'], **{name: sha(ROOT/name) for name in OWN}}


def simulate(case, policy):
    request = 500+round(case.ramp_seconds/DT)
    delay, stride = round(case.delay_seconds/DT), round(case.control_seconds/DT)
    commands, states = [], []
    target = base_motors = controller = hold_target = None
    hold_index = None
    with QuadrotorPhysics() as physics:
        physics.reset((0., 0., 3.5))
        states.append(physics.state())
        for i in range(request+4000):
            state = states[-1]
            if i == request:
                anchor = tuple(state['position'])
                if policy == 'attitude':
                    controller = AttitudeStop(state)
            info = None
            if i < 500:
                phase, relative, target = 'hover', i, (0., 0., 3.5)
            elif i < request:
                phase, relative = 'ramp', i-500
                distance = case.speed*(relative*DT)
                target = (distance*case.direction[0], distance*case.direction[1], 3.5)
            elif i < request+delay:
                phase, relative, target = 'delay_last_motors', i-request, None
            else:
                phase, relative = 'stop', i-request-delay
                if policy == 'original':
                    target = anchor
            updated = phase != 'delay_last_motors' and relative % stride == 0
            if updated:
                if phase == 'stop' and policy == 'attitude':
                    base_motors, info = controller.command(state)
                    target = info['target']
                    if controller.hold_target is not None and hold_index is None:
                        hold_target, hold_index = controller.hold_target, i
                else:
                    base_motors, inner = control(state, target)
                    info = dict(mode='ORIGINAL_CONTROL', saturated=inner['saturated'])
                    if phase == 'stop' and hold_index is None:
                        hold_target, hold_index = anchor, i
            if phase == 'delay_last_motors':
                applied = list(commands[-1]['motors'])
            else:
                applied = list(base_motors)
            pulse = case.name == 'newpitchpulse' and request-50 <= i < request
            if pulse:
                applied = [max(0., min(8., m+delta)) for m, delta in zip(applied, (-.12, -.12, .12, .12))]
            commands.append(dict(phase=phase, target=target, updated=updated, motors=applied, info=info, pulse=pulse))
            physics.step(applied)
            states.append(physics.state())
        metrics = analyze(states, request, case.direction)
        times = [s['time_s']-states[request]['time_s'] for s in states[request:]]
        flags = [hold_index is not None and i >= hold_index and hypot(*s['velocity']) <= .03
                 and hypot(*(p-a for p, a in zip(s['position'], hold_target))) <= .03
                 and abs(s['position'][2]-hold_target[2]) <= .1
                 for i, s in enumerate(states) if i >= request]
        stop = dict(hold_target=hold_target,
                    hold_latched_at_s=None if hold_index is None else states[hold_index]['time_s']-states[request]['time_s'],
                    actual_hold_window=stable_window(times, flags),
                    acceleration_clip_updates=sum(c['info'] is not None and c['info'].get('mode') == 'ATTITUDE_BRAKE'
                                                  and c['info']['acceleration_clipped'] for c in commands[request:]),
                    saturated_updates=sum(c['info'] is not None and c['info']['saturated'] for c in commands[request:]))
        result = dict(case=asdict(case), policy=policy, request_index=request, initial_position=[0., 0., 3.5],
                      dt_s=DT, states=states, commands=commands, metrics=metrics, stop=stop,
                      model_sha256=physics.model_sha256, dll_sha256=physics.dll_sha256,
                      engine_version=physics.version, flight_authorized=False)
    return canonical(result)


def independent_hold(raw):
    # Group consecutive valid indices instead of using the production rolling window.
    from itertools import groupby
    import math
    anchor = raw['stop']['hold_target']
    latch = raw['stop']['hold_latched_at_s']
    request = raw['request_index']
    states = raw['states'][request:]
    times = [s['time_s']-states[0]['time_s'] for s in states]
    valid = [anchor is not None and t >= latch-1e-9 and math.sqrt(sum(v*v for v in s['velocity'])) <= .03
             and math.sqrt(sum((p-a)**2 for p, a in zip(s['position'], anchor))) <= .03
             and abs(s['position'][2]-anchor[2]) <= .1 for t, s in zip(times, states)]
    groups = [list(items) for key, items in groupby(range(len(valid)), key=lambda i: valid[i]) if key]
    qualified = [g for g in groups if times[g[-1]]-times[g[0]] >= .3-1e-9]
    start = None if not qualified else times[qualified[0][0]]
    confirm = next((t for t in times if start is not None and t-start >= .3-1e-9), None)
    tail = groups[-1] if groups and groups[-1][-1] == len(valid)-1 else None
    result = dict(first_window_start_s=start, first_confirmation_s=confirm,
                  left_after_confirmation=confirm is not None and any(not flag for t, flag in zip(times, valid) if t > confirm),
                  final_continuous_start_s=None if tail is None else times[tail[0]],
                  final_window_confirmed=tail is not None and times[tail[-1]]-times[tail[0]] >= .3-1e-9)
    if result != raw['stop']['actual_hold_window']:
        raise ValueError('independent hold interval calculation differs')
    return True


def worker(name, policy, evidence, verify):
    case = next(c for c in ALL_CASES if c.name == name)
    raw = simulate(case, policy)
    audit = independent_metrics(raw)
    independent_hold(raw)
    old_matched = None
    if policy == 'original' and name in {c.name for c in CASES}:
        old = read(PARENT/name/'evidence.json.gz')
        old_matched = raw['states'] == old['states'] and [c['motors'] for c in raw['commands']] == [c['motors'] for c in old['commands']]
        if not old_matched:
            raise ValueError('original policy no longer matches frozen baseline')
    if verify:
        if raw != read(evidence):
            raise ValueError('feedback replay differs')
        with QuadrotorPhysics() as physics:
            physics.reset(raw['initial_position'])
            if physics.state() != raw['states'][0]:
                raise ValueError('initial replay mismatch')
            for i, command in enumerate(raw['commands'], 1):
                physics.step(command['motors'])
                if physics.state() != raw['states'][i]:
                    raise ValueError(f'motor replay mismatch at {i}')
    else:
        with evidence.open('xb') as stream:
            stream.write(gzip.compress(json.dumps(raw, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    return dict(name=name, policy=policy, verified=verify, steps=len(raw['commands']),
                independent_metrics=audit, independent_hold=True, old_baseline_matched=old_matched)


def make_chart(pair, field, title, unit, reference):
    series = [trial['points'] for trial in pair['trials']]
    high = max(max(p[field] for p in values) for values in series)
    low = min(0., min(min(p[field] for p in values) for values in series))
    high = max(high, reference, .001)
    high += (high-low)*.15
    y = lambda value: 225-(value-low)/(high-low)*185
    curves = []
    for points, color in zip(series, ('#bc6a36', '#126f88')):
        line = ' '.join(f'{60+p[0]/8*680:.2f},{y(p[field]):.2f}' for p in points)
        curves.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    return f'''<figure><figcaption>{title} · 橙：原位置保持；蓝：姿态反馈停止</figcaption><svg viewBox="0 0 790 265" role="img" aria-label="{title}">
    <path d="M60 40V225H740" fill="none" stroke="#789"/><line x1="60" x2="740" y1="{y(reference):.2f}" y2="{y(reference):.2f}" stroke="#8a929c" stroke-dasharray="5 4"/>
    {''.join(curves)}<text x="4" y="43">{high:.3f}</text><text x="4" y="225">{low:.3f}</text><text x="65" y="24">{unit}；虚线对照 {reference:.4f}</text><text x="60" y="251">请求后 0 秒</text><text x="695" y="251">8 秒</text></svg></figure>'''


def page(report):
    sections, rows, options = [], [], []
    fmt = lambda v: '未确认' if v is None else f'{v:.3f}'
    for i, pair in enumerate(report['pairs']):
        a, b = pair['trials']
        am, bm = a['metrics'], b['metrics']
        options.append(f'<option value="{i}">{pair["name"]} · {"原九组对照" if pair["baseline_case"] else "新增固定条件"}</option>')
        sections.append(f'''<section class="pair" data-pair="{i}" style="display:{'block' if i==0 else 'none'}"><h2>{pair['name']}：同一运动状态，两种停止方式</h2>
        <p>实际入口速度 {am['entry_horizontal_speed_mps']:.4f} m/s；停止请求前反馈与实际推力完全一致。</p>
        <p>最大前冲：原策略 <b>{am['max_forward_m']:.4f} m</b> → 新策略 <b>{bm['max_forward_m']:.4f} m</b>。低速持续确认：<b>{fmt(am['speed_window']['first_confirmation_s'])} s</b> → <b>{fmt(bm['speed_window']['first_confirmation_s'])} s</b>。</p>
        <p>新策略保持点锁定于请求后 {fmt(b['stop']['hold_latched_at_s'])} 秒；新目标稳定确认 {fmt(b['stop']['actual_hold_window']['first_confirmation_s'])} 秒。与原请求位置的偏移是允许的，不能只看降速更快就判定更安全。</p>
        {make_chart(pair, 1, '三维速度', 'm/s', .03)}{make_chart(pair, 2, '沿入口方向的位置', 'm', am['ideal_distance_m'])}
        <details><summary>查看完整指标、位置保持和限幅</summary><pre>{html.escape(json.dumps(dict(original=dict(metrics=am, stop=a['stop']), candidate=dict(metrics=bm, stop=b['stop'])), ensure_ascii=False, indent=2))}</pre></details></section>''')
        rows.append(f'<tr><td>{pair["name"]}</td><td>{am["max_forward_m"]:.4f} → {bm["max_forward_m"]:.4f}</td><td>{fmt(am["speed_window"]["first_confirmation_s"])} → {fmt(bm["speed_window"]["first_confirmation_s"])}</td><td>{am["horizontal_path_m"]:.4f} → {bm["horizontal_path_m"]:.4f}</td></tr>')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>停止策略物理对照</title><style>body{margin:0;padding:24px;background:#edf2f5;color:#193346;font:16px/1.7 system-ui}main{max-width:1080px;margin:auto}h1{font-size:30px}section{background:white;padding:22px;border-radius:12px;margin:20px 0}select{font:inherit;padding:10px;max-width:100%}figure{margin:20px 0}svg{width:100%}svg text{font-size:13px;fill:#586977}figcaption{font-weight:600}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dde5e8;text-align:left;white-space:nowrap}pre{overflow:auto;max-height:360px;font-size:13px}.note{color:#8b501e}a{color:#126f88}</style><main><h1>先减速，再选择保持位置</h1><section><p>原九组条件 + 六组新增条件 · 每组两种策略 · 原始推力和状态完整保存</p><p class="note">固定 MuJoCo 机体的开发对照，不是真实飞行、视觉导航或停止范围认证。新策略允许停在前方，必须同时比较时间和距离；参数在运行前固定，未根据结果筛选成功场景。</p><label for="pair">选择条件：</label><select id="pair">'''+''.join(options)+'''</select></section>'''+''.join(sections)+'''<section><h2>全部对照（原策略 → 新策略）</h2><div class="scroll"><table><tr><th>条件</th><th>最大前冲 m</th><th>持续低速确认 s</th><th>累计水平路程 m</th></tr>'''+''.join(rows)+'''</table></div><p>低速：三维速度 ≤ 0.03 m/s，连续 0.30 秒。首次确认后仍观察到 8 秒。曲线每 20 ms 取点，指标按每个 2 ms 状态计算。入口已低速的案例单列于完整报告。</p></section><section><h2>可复核记录</h2><p><a href="report.json">完整对照报告</a> · <a href="protocol.md">固定方案与推导</a></p><p>原位置附近的 hold_window 仅作为历史指标保留。actual_hold_window 使用各策略实际锁定的保持目标，且从锁定后才开始计时。</p></section></main><script>document.getElementById('pair').addEventListener('change',function(){document.querySelectorAll('.pair').forEach(e=>e.style.display=e.dataset.pair===this.value?'block':'none')})</script></html>'''


def run(output, verify):
    fixed = source_hashes()
    if verify:
        count = check_manifest(output)
        saved = json.loads((output/'report.json').read_text(encoding='utf-8'))
        if saved['sources'] != fixed or saved['parent_sha256'] != sha(PARENT/'report.json'):
            raise ValueError('source or parent changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'protocol.md').write_bytes((ROOT/'docs/STOPPING_POLICY_PROTOCOL.md').read_bytes())
    pairs = []
    for case in ALL_CASES:
        trials, evidence_pair = [], []
        for policy in ('original', 'attitude'):
            folder = output/case.name/policy
            if not verify: folder.mkdir(parents=True)
            path = folder/'evidence.json.gz'
            command = [sys.executable, '-X', 'faulthandler', str(Path(__file__).resolve()), '--worker', case.name,
                       '--policy', policy, '--evidence', str(path)]
            if verify: command.append('--verify')
            completed = subprocess.run(command, capture_output=True, encoding='utf-8', errors='replace', timeout=60)
            if not verify:
                (folder/'stdout.log').write_text(completed.stdout, encoding='utf-8')
                (folder/'stderr.log').write_text(completed.stderr, encoding='utf-8')
            if completed.returncode:
                print(completed.stdout, completed.stderr, flush=True)
                raise RuntimeError(f'{case.name}/{policy} failed: {completed.returncode}; no retry')
            audit = json.loads(completed.stdout)
            raw = read(path)
            evidence_pair.append(raw)
            entry = raw['states'][raw['request_index']]
            axis = raw['metrics']['evaluation_direction']
            points = [[s['time_s']-entry['time_s'], hypot(*s['velocity']),
                       sum((s['position'][j]-entry['position'][j])*axis[j] for j in range(2))]
                      for s in raw['states'][raw['request_index']::10]]
            trials.append(dict(policy=policy, input=f'{case.name}/{policy}/evidence.json.gz',
                               steps=len(raw['commands']), metrics=raw['metrics'], stop=raw['stop'], points=points,
                               independent_metrics=audit['independent_metrics'], independent_hold=audit['independent_hold'],
                               old_baseline_matched=audit['old_baseline_matched'], model_sha256=raw['model_sha256'],
                               dll_sha256=raw['dll_sha256']))
        a, b = evidence_pair
        end = a['request_index']+round(case.delay_seconds/DT)
        same_prefix = a['states'][:end+1] == b['states'][:end+1] and [c['motors'] for c in a['commands'][:end]] == [c['motors'] for c in b['commands'][:end]]
        if not same_prefix: raise ValueError('paired entry or delay trajectory differs')
        pairs.append(dict(name=case.name, spec=asdict(case), baseline_case=case in CASES,
                          prefix_matched=True, trials=trials))
        am, bm = a['metrics'], b['metrics']
        print(case.name, 'verified' if verify else 'recorded',
              'distance', round(am['max_forward_m'], 4), '->', round(bm['max_forward_m'], 4),
              'slow', am['speed_window']['first_confirmation_s'], '->', bm['speed_window']['first_confirmation_s'], flush=True)
    if fixed != source_hashes(): raise ValueError('source changed during run')
    def faster(pair):
        a, b = [t['metrics']['speed_window']['first_confirmation_s'] for t in pair['trials']]
        return b is not None and (a is None or b < a-1e-9)
    summary = dict(pairs=len(pairs), trials=len(pairs)*2, physics_steps=sum(t['steps'] for p in pairs for t in p['trials']),
                   candidate_faster_slow_confirmation=sum(faster(p) for p in pairs),
                   candidate_larger_forward=sum(p['trials'][1]['metrics']['max_forward_m'] > p['trials'][0]['metrics']['max_forward_m']+1e-9 for p in pairs),
                   candidate_final_hold_confirmed=sum(p['trials'][1]['stop']['actual_hold_window']['final_window_confirmed'] for p in pairs),
                   candidate_slow_from_ideal_deadline=sum(p['trials'][1]['metrics']['slow_from_ideal_deadline'] for p in pairs))
    report = canonical(dict(sources=fixed, parent_sha256=sha(PARENT/'report.json'), pairs=pairs, summary=summary,
                            new_model_calls=0, real_flights=0, flight_authorized=False))
    if verify:
        if report != saved or page(report) != (output/'demo.html').read_text(encoding='utf-8'):
            raise ValueError('report/page replay differs')
        if (output/'protocol.md').read_bytes() != (ROOT/'docs/STOPPING_POLICY_PROTOCOL.md').read_bytes() or check_manifest(output) != count:
            raise ValueError('protocol or archive changed')
    else:
        dump(output/'report.json', report)
        (output/'demo.html').write_text(page(report), encoding='utf-8')
        dump(output/'manifest.json', {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()})
        count = check_manifest(output)
    return dict(verified=verify, files=count, sources=len(fixed), **summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--worker', choices=[c.name for c in ALL_CASES])
    parser.add_argument('--policy', choices=('original', 'attitude'))
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if args.worker:
        if not args.policy or args.evidence is None: parser.error('--policy and --evidence required')
        result = worker(args.worker, args.policy, args.evidence.resolve(), args.verify)
    else:
        if args.output is None: parser.error('--output required')
        result = run(args.output.resolve(), args.verify)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
