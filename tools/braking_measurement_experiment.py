"""来源：本项目原创。独立进程运行、反馈重算、电机重放及停止指标复核。"""
import argparse
from dataclasses import asdict
import gzip
import html
from itertools import groupby
import json
from math import sqrt
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drone_nav.braking_measurement import CASES, simulate
from drone_nav.realvision import sha, dump
from tools.object_probe import check_manifest

PARENT = ROOT/'work/rolling-guard-02'
PROTOCOL = ROOT/'docs/BRAKING_MEASUREMENT_PROTOCOL.md'
OWN = ('drone_nav/braking_measurement.py', 'tools/braking_measurement_experiment.py',
       'tests/test_braking_measurement.py', 'docs/BRAKING_MEASUREMENT_PROTOCOL.md',
       'drone_nav/native_physics.py', 'drone_nav/flight_control.py', 'drone_nav/quadrotor.xml')
LABELS = dict(x05='+X · 0.05 m/s', x10='+X · 0.10 m/s', x20='+X · 0.20 m/s',
              y20='+Y · 0.20 m/s', diagonal20='对角线 · 0.20 m/s', negative20='-X · 0.20 m/s',
              early20='加速早期请求停止', delayed20='推力保持延迟 0.2 秒', control20ms='每 20 ms 更新控制')


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def sources():
    check_manifest(PARENT)
    previous = json.loads((PARENT/'report.json').read_text(encoding='utf-8'))['sources']
    for path, digest in previous.items():
        if sha(ROOT/path) != digest:
            raise ValueError('historical source changed: '+path)
    return {**previous, **{path: sha(ROOT/path) for path in OWN}}


def independent_metrics(raw):
    """Aggregate raw states separately: contiguous index groups, direct vector projection."""
    states = raw['states'][raw['request_index']:]
    entry = states[0]
    times = [s['time_s']-entry['time_s'] for s in states]
    norm = lambda values: sqrt(sum(v*v for v in values))
    v = norm(entry['velocity'][:2])
    axis = [q/v for q in entry['velocity'][:2]] if v > 1e-9 else raw['case']['direction']
    xyz = [[s['position'][i]-entry['position'][i] for i in range(3)] for s in states]
    forward = [sum(p[i]*axis[i] for i in range(2)) for p in xyz]
    speed = [norm(s['velocity']) for s in states]
    slow = [s <= .03 for s in speed]
    hold = [slow[i] and norm(p) <= .03 and abs(p[2]) <= .1 for i, p in enumerate(xyz)]

    def windows(flags):
        ranges = []
        for valid, indices in groupby(range(len(flags)), key=lambda i: flags[i]):
            indices = list(indices)
            if valid:
                ranges.append((indices[0], indices[-1]))
        qualified = [(a, b) for a, b in ranges if times[b]-times[a] >= .3-1e-9]
        first_start = times[qualified[0][0]] if qualified else None
        confirmation = next((t for t in times if first_start is not None and t-first_start >= .3-1e-9), None)
        tail = ranges[-1] if ranges and ranges[-1][1] == len(flags)-1 else None
        return dict(first_window_start_s=first_start, first_confirmation_s=confirmation,
                    left_after_confirmation=confirmation is not None and any(not ok for t, ok in zip(times, flags) if t > confirmation),
                    final_continuous_start_s=None if tail is None else times[tail[0]],
                    final_window_confirmed=tail is not None and times[tail[1]]-times[tail[0]] >= .3-1e-9)

    distance, deadline = v*.2+v*v/2, .2+v
    index = next(i for i, t in enumerate(times) if t >= deadline-1e-9)
    measured = dict(entry_horizontal_speed_mps=v, max_forward_m=max(forward),
                    max_backward_m=max(0., -min(forward)),
                    max_lateral_m=max(abs(p[0]*axis[1]-p[1]*axis[0]) for p in xyz),
                    max_horizontal_radius_m=max(norm(p[:2]) for p in xyz),
                    horizontal_path_m=sum(norm([b[i]-a[i] for i in range(2)]) for a, b in zip(xyz, xyz[1:])),
                    max_height_change_m=max(abs(p[2]) for p in xyz),
                    ideal_distance_m=distance, ideal_stop_time_s=deadline,
                    speed_at_ideal_deadline_mps=speed[index],
                    exceeds_ideal_distance=max(forward) > distance+1e-9,
                    exceeds_ideal_distance_plus_margin=max(forward) > distance+.05+1e-9,
                    slow_from_ideal_deadline=all(slow[index:]),
                    speed_window=windows(slow), hold_window=windows(hold))

    def compare(a, b):
        if isinstance(a, dict):
            return all(compare(value, b[key]) for key, value in a.items())
        if type(a) is float and type(b) in (int, float):
            return abs(a-b) <= 1e-10
        return a == b

    if not compare(measured, raw['metrics']):
        raise ValueError('independent metric calculation differs')
    return dict(passed=True, sampled_states=len(states), independently_checked_keys=list(measured))


def read_evidence(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def worker(case_name, evidence, verify):
    case = next(c for c in CASES if c.name == case_name)
    actual = canonical(simulate(case))
    metrics_audit = independent_metrics(actual)
    if verify:
        saved = read_evidence(evidence)
        if actual != saved:
            raise ValueError('fresh feedback-control replay differs')
        # Independent actuator replay bypasses the position controller entirely.
        from drone_nav.native_physics import QuadrotorPhysics
        with QuadrotorPhysics() as physics:
            physics.reset(saved['initial_position'])
            if physics.state() != saved['states'][0]:
                raise ValueError('initial motor replay state differs')
            for i, command in enumerate(saved['commands'], 1):
                physics.step(command['motors'])
                if physics.state() != saved['states'][i]:
                    raise ValueError(f'motor replay state differs at {i}')
    else:
        with evidence.open('xb') as stream:
            stream.write(gzip.compress(json.dumps(actual, separators=(',', ':'), allow_nan=False).encode('utf-8'), mtime=0))
    return dict(case=case_name, verified=verify, steps=len(actual['commands']),
                independent_metrics=metrics_audit, feedback_replay=verify, motor_replay=verify)


def chart(points, values, reference, deadline, title, unit):
    low, high = min(0., min(values)), max(max(values), reference, .001)
    high += (high-low)*.15
    scale = lambda value: 220-(value-low)/(high-low)*180
    x = lambda time: 55+time/8*680
    line = ' '.join(f'{x(p[0]):.2f},{scale(v):.2f}' for p, v in zip(points, values))
    return f'''<figure><figcaption>{title}</figcaption><svg viewBox="0 0 780 265" role="img" aria-label="{title}">
    <path d="M55 30V220H735" fill="none" stroke="#789"/><line x1="55" y1="{scale(reference):.2f}" x2="735" y2="{scale(reference):.2f}" stroke="#b46534" stroke-dasharray="5 4"/>
    <line x1="{x(deadline):.2f}" y1="30" x2="{x(deadline):.2f}" y2="220" stroke="#8d719c" stroke-dasharray="5 4"/>
    <polyline points="{line}" fill="none" stroke="#167487" stroke-width="2.5"/>
    <text x="5" y="40">{high:.3f}</text><text x="5" y="220">{low:.3f}</text><text x="55" y="249">请求后 0 秒</text><text x="685" y="249">8 秒</text><text x="60" y="22">{unit} · 棕虚线为对照值，紫虚线为理想停止期限</text></svg></figure>'''


def page(report):
    panels, rows, options = [], [], []
    for i, case in enumerate(report['cases']):
        name, m, points = case['name'], case['metrics'], case['points']
        options.append(f'<option value="{i}">{LABELS[name]}</option>')
        low = m['speed_window']['first_confirmation_s']
        hold = m['hold_window']['first_confirmation_s']
        fmt = lambda value: '未确认' if value is None else f'{value:.3f}'
        verdict = '超过理想距离' if m['exceeds_ideal_distance'] else '未超过理想距离'
        panels.append(f'''<section class="case" data-case="{i}" style="display:{'block' if i==0 else 'none'}"><h2>{LABELS[name]}</h2>
        <p>实际入口水平速度 <b>{m['entry_horizontal_speed_mps']:.4f} m/s</b> · 最大前冲 <b>{m['max_forward_m']:.4f} m</b> · 理想对照 {m['ideal_distance_m']:.4f} m</p>
        <p>首次低速持续 0.30 秒确认：{fmt(low)} 秒；位置保持稳定确认：{fmt(hold)} 秒。理想停止期限 {m['ideal_stop_time_s']:.3f} 秒时，三维速度仍为 {m['speed_at_ideal_deadline_mps']:.4f} m/s。</p>
        <p>{verdict}；额外 0.05 米余量：{'仍超过' if m['exceeds_ideal_distance_plus_margin'] else '未超过'}。入口已低于速度容差：{'是' if m['entry_already_slow'] else '否'}。</p>
        {chart(points, [p[1] for p in points], .03, m['ideal_stop_time_s'], '三维速度随时间变化', 'm/s')}
        {chart(points, [p[2] for p in points], m['ideal_distance_m'], m['ideal_stop_time_s'], '沿入口方向的位置变化', 'm')}
        <details><summary>完整指标、回摆和持续稳定窗口</summary><pre>{html.escape(json.dumps(m, ensure_ascii=False, indent=2))}</pre></details></section>''')
        rows.append(f'<tr><td>{LABELS[name]}</td><td>{m["entry_horizontal_speed_mps"]:.4f}</td><td>{m["max_forward_m"]:.4f}</td><td>{m["ideal_distance_m"]:.4f}</td><td>{fmt(hold)}</td><td>{"是" if m["slow_from_ideal_deadline"] else "否"}</td></tr>')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>四旋翼低速停止响应</title><style>body{margin:0;background:#edf2f5;color:#193346;font:16px/1.7 system-ui;padding:24px}main{max-width:1080px;margin:auto}h1{font-size:30px}section{background:white;border-radius:12px;padding:22px;margin:20px 0}select{font:inherit;padding:10px;max-width:100%}svg{width:100%}figure{margin:20px 0}figcaption{font-weight:bold}svg text{font-size:13px;fill:#536574}pre{overflow:auto;max-height:340px;font-size:13px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #dde5e8;white-space:nowrap}a{color:#126c83}.note{color:#8b501e}</style><main><h1>从“假定能刹住”到测量实际仿真响应</h1><section><p>原控制器 · MuJoCo 3.3.7 · 每 2 ms 保存反馈 · 九组固定条件</p><p class="note">这是理想机体的物理仿真测量，不是真实飞行、视觉避障或制动能力认证。名义目标速度不等于实际入口速度；不把首次低速视为永久停稳。</p><p>停止请求后保持当时实际位置，观察 8 秒。对照为反应 0.2 秒、减速度 1 m/s² 的一维模型；单独展示 0.05 米额外余量的影响。</p><label for="case">选择条件：</label><select id="case">'''+''.join(options)+'''</select></section>'''+''.join(panels)+'''<section><h2>九组结果</h2><div class="scroll"><table><tr><th>条件</th><th>入口速度 m/s</th><th>最大前冲 m</th><th>理想距离 m</th><th>位置保持确认 s</th><th>理想期限后持续低速</th></tr>'''+''.join(rows)+'''</table></div><p>确认条件：位置误差 ≤ 0.03 米、三维速度 ≤ 0.03 m/s、高度误差 ≤ 0.10 米，连续 0.30 秒。所有指标按完整 2 ms 记录计算；曲线仅每 20 ms 取点显示。</p></section><section><h2>复核材料</h2><p><a href="report.json">报告与来源摘要</a> · <a href="protocol.md">固定实验方案</a></p><p>完整电机命令和反馈保存在各案例 evidence.json.gz 中。新进程反馈重算、电机单独重放和独立指标聚合在验证命令中执行；本页展示保存数据。</p></section></main><script>document.getElementById('case').addEventListener('change',function(){document.querySelectorAll('.case').forEach(e=>e.style.display=e.dataset.case===this.value?'block':'none')})</script></html>'''


def execute(output, verify=False):
    fixed = sources()
    expected = None
    if verify:
        count = check_manifest(output)
        expected = json.loads((output/'report.json').read_text(encoding='utf-8'))
        if expected['sources'] != fixed or expected['parent_sha256'] != sha(PARENT/'report.json'):
            raise ValueError('source or parent mismatch')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'protocol.md').write_bytes(PROTOCOL.read_bytes())
    cases = []
    for case in CASES:
        folder = output/case.name
        if not verify:
            folder.mkdir()
        evidence = folder/'evidence.json.gz'
        args = [sys.executable, '-X', 'faulthandler', str(Path(__file__).resolve()),
                '--worker', case.name, '--evidence', str(evidence)]
        if verify:
            args.append('--verify')
        completed = subprocess.run(args, capture_output=True, encoding='utf-8', errors='replace', timeout=60)
        if not verify:
            (folder/'stdout.log').write_text(completed.stdout, encoding='utf-8')
            (folder/'stderr.log').write_text(completed.stderr, encoding='utf-8')
        if completed.returncode:
            print(completed.stdout, completed.stderr, flush=True)
            raise RuntimeError(f'{case.name} child failed with exit code {completed.returncode}; no retry')
        audit = json.loads(completed.stdout)
        if audit['case'] != case.name or not audit['independent_metrics']['passed']:
            raise ValueError('invalid child audit')
        raw = read_evidence(evidence)
        if raw['case'] != canonical(asdict(case)):
            raise ValueError('saved case mismatch')
        entry = raw['states'][raw['request_index']]
        selected = raw['states'][raw['request_index']::10]
        if selected[-1] != raw['states'][-1]:
            selected.append(raw['states'][-1])
        points = [[s['time_s']-entry['time_s'], sqrt(sum(v*v for v in s['velocity'])),
                   sum((s['position'][j]-entry['position'][j])*raw['metrics']['evaluation_direction'][j] for j in range(2))]
                  for s in selected]
        cases.append(dict(name=case.name, spec=raw['case'], input=f'{case.name}/evidence.json.gz',
                          model_sha256=raw['model_sha256'], dll_sha256=raw['dll_sha256'],
                          engine_version=raw['engine_version'], steps=len(raw['commands']), metrics=raw['metrics'],
                          points=points, independent_metrics=audit['independent_metrics']))
        print(case.name, 'verified' if verify else 'recorded', 'entry', round(raw['metrics']['entry_horizontal_speed_mps'], 4),
              'max_forward', round(raw['metrics']['max_forward_m'], 4), flush=True)
    if sources() != fixed:
        raise ValueError('sources changed during experiment')
    summary = dict(cases=len(cases), physics_steps=sum(c['steps'] for c in cases),
                   exceeds_ideal_distance=sum(c['metrics']['exceeds_ideal_distance'] for c in cases),
                   exceeds_ideal_distance_plus_margin=sum(c['metrics']['exceeds_ideal_distance_plus_margin'] for c in cases),
                   slow_from_ideal_deadline=sum(c['metrics']['slow_from_ideal_deadline'] for c in cases),
                   final_hold_confirmed=sum(c['metrics']['hold_window']['final_window_confirmed'] for c in cases),
                   entry_already_slow=sum(c['metrics']['entry_already_slow'] for c in cases))
    report = dict(sources=fixed, parent_sha256=sha(PARENT/'report.json'), scope='MuJoCo fixed-model stopping measurement',
                  new_model_calls=0, real_flights=0, flight_authorized=False, cases=cases, summary=summary)
    if verify:
        if report != expected or page(report) != (output/'demo.html').read_text(encoding='utf-8'):
            raise ValueError('report or page reconstruction differs')
        if (output/'protocol.md').read_bytes() != PROTOCOL.read_bytes() or check_manifest(output) != count:
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
    parser.add_argument('--worker', choices=[c.name for c in CASES])
    parser.add_argument('--evidence', type=Path)
    options = parser.parse_args()
    if options.worker:
        if options.evidence is None:
            parser.error('--worker requires --evidence')
        result = worker(options.worker, options.evidence.resolve(), options.verify)
    else:
        if options.output is None:
            parser.error('--output required')
        result = execute(options.output.resolve(), options.verify)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
