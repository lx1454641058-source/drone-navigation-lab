"""来源：本项目原创。独立预测进程与壁钟调度的停止反馈；记录并重放实际事件。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import html
import json
from math import sqrt
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drone_nav.async_stop import (make_request, request_record, StopStepper, checkpoint, predict_index,
    summarize_trace, ForecastMonitor, HORIZON_STEPS, TICK_STEPS, TICK_SECONDS, RESULT_DEADLINE)
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.stop_forecast import FrozenForecast, StopRecipe
from drone_nav.realvision import sha, dump
from tools.stopping_policy_experiment import read, independent_hold
from tools.braking_measurement_experiment import independent_metrics, canonical
from tools.object_probe import check_manifest

PARENT = ROOT/'work/stop-forecast-02'
OWN = ('drone_nav/async_stop.py', 'tools/async_stop_experiment.py',
       'tests/test_async_stop.py', 'docs/ASYNC_STOP_PROTOCOL.md')
TRIALS = (
    ('original-normal', 'original', 'normal'),
    ('attitude-normal', 'attitude', 'normal'),
    ('late-result', 'attitude', 'late'),
    ('worker-error', 'attitude', 'worker-error'),
    ('wrong-request', 'attitude', 'wrong-request'),
    ('early-pitch', 'attitude', 'early-pitch'),
    ('late-thrust', 'attitude', 'late-thrust'),
    ('control-stall', 'attitude', 'control-stall'),
)


def sources():
    check_manifest(PARENT)
    previous = json.loads((PARENT/'report.json').read_text(encoding='utf-8'))['sources']
    for name, value in previous.items():
        if sha(ROOT/name) != value:
            raise ValueError('frozen source changed: '+name)
    return {**previous, **{name:sha(ROOT/name) for name in OWN}}


def write_gzip(path, value):
    with path.open('xb') as stream:
        stream.write(gzip.compress(json.dumps(value, separators=(',', ':'), allow_nan=False).encode(), mtime=0))


def ready():
    import faulthandler
    faulthandler.enable()
    with QuadrotorPhysics() as physics:
        return dict(pid=os.getpid(), version=physics.version)


def mutate_packet(packet, mode):
    if mode == 'wrong-request':
        record = packet.record()
        record['request_id'] += '-other'
        return FrozenForecast.create(record)
    return packet


def predict_job(request, folder, mode):
    started = perf_counter()
    if mode == 'worker-error':
        dump(folder/'worker.json', dict(pid=os.getpid(), error='injected prediction worker failure'))
        raise RuntimeError('injected prediction worker failure')
    packet, trace = predict_index(request)
    calculated = perf_counter()-started
    packet = mutate_packet(packet, mode)
    write_gzip(folder/'prediction.json.gz', trace)
    dump(folder/'packet.json', asdict(packet))
    if mode == 'late':
        sleep(.60)
    metadata = dict(pid=os.getpid(), prediction_seconds=calculated,
                    job_before_final_metadata_write_seconds=perf_counter()-started,
                    packet_bytes=len(packet.content.encode('utf-8')), injected_wait_seconds=.60 if mode == 'late' else 0.)
    dump(folder/'worker.json', metadata)
    return packet, metadata


def fault_command(command, index, mode):
    command = canonical(command)
    if mode == 'early-pitch' and index < 100:
        command['motors'] = [max(0., min(8., m+d)) for m,d in zip(command['motors'], (-.25, -.25, .25, .25))]
    if mode == 'late-thrust' and 500 <= index < 600:
        command['motors'] = [m*.85 for m in command['motors']]
    return command


def prepare(physics, policy):
    parent_path = ROOT/'work/stopping-policy-01/control20ms'/policy/'evidence.json.gz'
    parent = read(parent_path)
    physics.reset(parent['initial_position'])
    for command in parent['commands'][:parent['request_index']]:
        physics.step(command['motors'])
    if physics.state() != parent['states'][parent['request_index']]:
        raise ValueError('entry reconstruction differs')
    return parent_path, parent


def run_trial(spec, folder):
    name, policy, mode = spec
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as executor:
        worker = executor.submit(ready).result(timeout=20)
        if worker['pid'] == os.getpid():
            raise ValueError('prediction worker is not a separate process')
        with QuadrotorPhysics() as physics:
            parent_path, parent = prepare(physics, policy)
            request = make_request(physics, StopRecipe(policy, control_seconds=.02), name)
            record, recipe = request_record(request)
            stepper = StopStepper(record['entry'], recipe, record['prior_motors'])
            monitor = ForecastMonitor(request)
            states, commands, events = [physics.state()], [], []
            received = False
            packet = metadata = worker_error_text = None
            previous_missed = False
            started = perf_counter()
            future = executor.submit(predict_job, request, folder, mode)
            for tick in range(HORIZON_STEPS//TICK_STEPS+1):
                scheduled = tick*TICK_SECONDS
                remaining = started+scheduled-perf_counter()
                if remaining > 0:
                    sleep(remaining)
                if mode == 'control-stall' and tick == 50:
                    sleep(.06)
                begin = perf_counter()-started
                current = checkpoint(physics, stepper)
                incoming = None
                error = False
                if not received and future.done():
                    received = True
                    try:
                        packet, metadata = future.result()
                        incoming = packet
                    except Exception as exc:
                        worker_error_text = f'{type(exc).__name__}: {exc}'
                        error = True
                elapsed = perf_counter()-started
                missed_input = previous_missed or begin-scheduled > TICK_SECONDS
                decision = monitor.tick(current, elapsed_s=elapsed, packet=incoming,
                                        worker_error=error, deadline_missed=missed_input)
                event = dict(tick=tick, scheduled_s=scheduled, begin_s=begin, monitor_elapsed_s=elapsed,
                    checkpoint=current, delivered=incoming is not None, worker_error=error,
                    deadline_missed_input=missed_input, decision=decision)
                if tick < HORIZON_STEPS//TICK_STEPS:
                    for _ in range(TICK_STEPS):
                        index = stepper.index
                        command = fault_command(stepper.command(states[-1]), index, mode)
                        physics.step(command['motors'])
                        commands.append(command)
                        states.append(physics.state())
                events.append(event)
                end = perf_counter()-started
                previous_missed = end > scheduled+TICK_SECONDS
                event.update(end_s=end, work_s=end-begin, start_lateness_s=max(0., begin-scheduled),
                             completed_after_next_deadline=previous_missed)
            if not received:
                raise RuntimeError('worker not terminal within the fixed 8 s run; outputs preserved')
            actual = summarize_trace(states, commands, stepper)
    raw = canonical(dict(name=name, policy=policy, mode=mode, request=asdict(request), actual=actual,
        events=events, parent_input=parent_path.relative_to(ROOT).as_posix(), parent_sha256=sha(parent_path),
        prefix_steps=parent['request_index'], main_pid=os.getpid(), ready_worker=worker,
        delivered_worker=metadata, worker_error=worker_error_text, simulation_clock_paused_for_prediction=False,
        wall_clock_paced=True, hard_realtime_certified=False, execution_authorized=False))
    if mode not in ('early-pitch', 'late-thrust'):
        if (states != parent['states'][parent['request_index']:]
            or [c['motors'] for c in commands] != [c['motors'] for c in parent['commands'][parent['request_index']:]]):
            raise ValueError('asynchronous result handling changed frozen feedback execution')
    independent_metrics(actual)
    independent_hold(actual)
    write_gzip(folder/'actual.json.gz', raw)
    return raw


def load_packet(folder):
    return FrozenForecast(**json.loads((folder/'packet.json').read_text(encoding='utf-8')))


def independent_bounds(packet, prediction):
    points = packet.record()['checkpoints']
    count = 0
    for point in points:
        tail = prediction['states'][point['index']:]
        minimum = [min(s['position'][j] for s in tail) for j in range(3)]
        maximum = [max(s['position'][j] for s in tail) for j in range(3)]
        if point['remaining']['center_lower_m'] != minimum or point['remaining']['center_upper_m'] != maximum:
            raise ValueError('independent suffix bounds differ')
        count += len(tail)
    return dict(checkpoints=len(points), suffix_sample_visits=count, includes_entire_remaining_horizon=True)


def verify_trial(spec, folder):
    name, policy, mode = spec
    raw = read(folder/'actual.json.gz')
    request = FrozenForecast(**raw['request'])
    if sha(ROOT/raw['parent_input']) != raw['parent_sha256']:
        raise ValueError('parent input changed')
    packet = None
    if mode != 'worker-error':
        packet = load_packet(folder)
        rebuilt_packet, prediction = predict_index(request)
        rebuilt_packet = mutate_packet(rebuilt_packet, mode)
        if packet != rebuilt_packet or prediction != read(folder/'prediction.json.gz'):
            raise ValueError('cross-process prediction does not rebuild')
        independent_metrics(prediction)
        independent_hold(prediction)
        independent_bounds(packet, prediction)
    monitor = ForecastMonitor(request)
    with QuadrotorPhysics() as physics:
        _, parent = prepare(physics, policy)
        regenerated_request = make_request(physics, StopRecipe(policy, control_seconds=.02), name)
        if regenerated_request != request:
            raise ValueError('saved request differs from actual prefix')
        record, recipe = request_record(request)
        stepper = StopStepper(record['entry'], recipe, record['prior_motors'])
        states, commands = [physics.state()], []
        for event in raw['events']:
            current = checkpoint(physics, stepper)
            if current != event['checkpoint']:
                raise ValueError('checkpoint or controller memory differs')
            result = monitor.tick(current, elapsed_s=event['monitor_elapsed_s'],
                packet=packet if event['delivered'] else None, worker_error=event['worker_error'],
                deadline_missed=event['deadline_missed_input'])
            if result != event['decision']:
                raise ValueError('monitor event replay differs')
            if event['tick'] < HORIZON_STEPS//TICK_STEPS:
                for _ in range(TICK_STEPS):
                    command = fault_command(stepper.command(states[-1]), stepper.index-1, mode)
                    physics.step(command['motors'])
                    commands.append(command)
                    states.append(physics.state())
        if summarize_trace(states, commands, stepper) != raw['actual']:
            raise ValueError('feedback replay differs')
    with QuadrotorPhysics() as physics:
        prepare(physics, policy)
        if physics.state() != raw['actual']['states'][0]:
            raise ValueError('motor replay entry differs')
        for i, command in enumerate(raw['actual']['commands'], 1):
            physics.step(command['motors'])
            if physics.state() != raw['actual']['states'][i]:
                raise ValueError('independent motor replay differs')
    independent_metrics(raw['actual'])
    independent_hold(raw['actual'])
    return raw


def summarize(spec, folder, raw):
    name, policy, mode = spec
    events = raw['events']
    if len(events) != 401 or [e['tick'] for e in events] != list(range(401)):
        raise ValueError('missing monitor ticks')
    previous_missed = False
    for event in events:
        if not event['scheduled_s'] <= event['begin_s'] <= event['monitor_elapsed_s'] <= event['end_s']:
            raise ValueError('invalid recorded clock order')
        missed = event['end_s'] > event['scheduled_s']+TICK_SECONDS
        if (missed != event['completed_after_next_deadline']
            or event['deadline_missed_input'] != (previous_missed or event['begin_s']-event['scheduled_s'] > TICK_SECONDS)
            or event['work_s'] != event['end_s']-event['begin_s']):
            raise ValueError('deadline evidence differs')
        previous_missed = missed
    worker = json.loads((folder/'worker.json').read_text(encoding='utf-8'))
    if worker['pid'] != raw['ready_worker']['pid'] or worker['pid'] == raw['main_pid']:
        raise ValueError('worker process evidence differs')
    receipt = next((e for e in events if e['delivered'] or e['worker_error']), None)
    matches = [e for e in events if e['decision']['forecast_matches']]
    failure = next((e for e in events if e['decision']['reason'] not in ('WAITING', 'MATCHED', 'HORIZON_EXHAUSTED')), None)
    prediction = None if mode == 'worker-error' else read(folder/'prediction.json.gz')
    audit = None if prediction is None else independent_bounds(load_packet(folder), prediction)
    error = None if prediction is None else max(sqrt(sum((a-b)**2 for a,b in zip(s['position'],p['position'])))
                                               for s,p in zip(raw['actual']['states'],prediction['states']))
    points = [[s['time_s']-raw['actual']['states'][0]['time_s'], s['position'][0]-raw['actual']['states'][0]['position'][0],
               None if prediction is None else prediction['states'][i]['position'][0]-prediction['states'][0]['position'][0]]
              for i,s in enumerate(raw['actual']['states']) if i % 10 == 0]
    return dict(name=name, policy=policy, mode=mode, worker=worker,
        receive_wall_s=None if receipt is None else receipt['monitor_elapsed_s'],
        receive_step=None if receipt is None else receipt['checkpoint']['index'],
        first_match_step=matches[0]['checkpoint']['index'] if matches else None, matching_ticks=len(matches),
        first_failure=None if failure is None else dict(reason=failure['decision']['reason'],
            step=failure['checkpoint']['index'], simulation_elapsed_s=failure['checkpoint']['index']*.002,
            wall_elapsed_s=failure['monitor_elapsed_s']),
        final_reason=events[-1]['decision']['reason'], max_work_s=max(e['work_s'] for e in events),
        median_work_s=statistics.median(e['work_s'] for e in events),
        max_start_lateness_s=max(e['start_lateness_s'] for e in events),
        missed_deadlines=sum(e['completed_after_next_deadline'] for e in events),
        max_position_error_m=error, independent_bounds=audit, actual_hold=raw['actual']['stop'],
        prefix_steps=raw['prefix_steps'], actual_steps=len(raw['actual']['commands']),
        prediction_steps=0 if prediction is None else len(prediction['commands']),
        points=points, cycle_ms=[e['work_s']*1000 for e in events], execution_authorized=False)


def plot(trial):
    points = trial['points']
    values = [v for p in points for v in p[1:] if v is not None]
    lower, upper = min(0., min(values)), max(.001, max(values))
    scale = upper-lower
    y = lambda v: 205-(v-lower)/scale*155
    paths = []
    for column, color in ((2, '#19798a'), (1, '#bd6730')):
        if points[0][column] is None: continue
        path = ' '.join(f'{55+p[0]/8*690:.2f},{y(p[column]):.2f}' for p in points)
        paths.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    return f'<svg viewBox="0 0 790 250" role="img" aria-label="预测与实际 X 方向位移"><path d="M55 35V205H745" stroke="#789" fill="none"/>{"".join(paths)}<text x="55" y="24">X 位移 m · 蓝：预测 · 橙：实际（重合时覆盖蓝线）</text><text x="2" y="55">{upper:.3f}</text><text x="2" y="205">{lower:.3f}</text><text x="55" y="235">0 秒</text><text x="705" y="235">8 秒</text></svg>'


def page(report):
    options, panels, rows = [], [], []
    for i,t in enumerate(report['trials']):
        failure = '无提前失效' if t['first_failure'] is None else t['first_failure']['reason']
        receive = '未返回' if t['receive_wall_s'] is None else f'{t["receive_wall_s"]*1000:.1f} ms / 已执行 {t["receive_step"]} 步'
        options.append(f'<option value="{i}">{t["name"]}</option>')
        panels.append(f'''<section class="trial" data-index="{i}" style="display:{'block' if i==0 else 'none'}"><h2>{t['name']}</h2>
        <p>结果/错误到达：<b>{receive}</b>；匹配周期：<b>{t['matching_ticks']}</b>。{failure}</p>
        <p>最大周期处理 {t['max_work_s']*1000:.2f} ms；错过下一计划时刻 <b>{t['missed_deadlines']}</b> 个周期。最终状态：{t['final_reason']}。</p>
        {plot(t)}<p>三维最大位置差：{t['max_position_error_m']} m。图只展示 X 方向，推力损失的高度误差需看此三维指标。</p>
        <details><summary>实际保持窗口与失效时刻</summary><pre>{html.escape(json.dumps(dict(actual_hold=t['actual_hold'],first_failure=t['first_failure'],worker=t['worker']),ensure_ascii=False,indent=2))}</pre></details></section>''')
        rows.append(f'<tr><td>{t["name"]}</td><td>{t["matching_ticks"]}</td><td>{failure}</td><td>{t["max_work_s"]*1000:.2f}</td><td>{t["missed_deadlines"]}</td></tr>')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>异步停止预测与持续反馈</title><style>body{margin:0;padding:24px;background:#edf2f5;color:#193346;font:16px/1.7 system-ui}main{max-width:1080px;margin:auto}section{padding:22px;margin:20px 0;background:white;border-radius:12px}h1{font-size:30px}select{font:inherit;padding:9px;max-width:100%}svg{width:100%}svg text{font-size:13px;fill:#586977}pre{max-height:360px;overflow:auto;font-size:13px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}.note{color:#89501e}a{color:#126f88}</style><main><h1>预测在后台计算，停止反馈持续执行</h1><section><p>20 ms 主循环 · 独立预测进程 · 当前物理状态和控制器记忆同时核对</p><p class="note">这是普通 Windows 调度的模型实验，不是硬实时系统或导航授权。收到预测不会恢复移动；迟到、偏差和控制超期锁定失效。完整停止反馈始终继续。</p><p>预测保留未来回摆及全部 2 ms 样本；有限轨迹包围不能代替视觉空闲或模型误差保证。</p><label for="trial">选择场景：</label><select id="trial">'''+''.join(options)+'''</select></section>'''+''.join(panels)+'''<section><h2>全部并发测量</h2><div class="scroll"><table><tr><th>场景</th><th>匹配周期</th><th>首次失效</th><th>最大周期处理 ms</th><th>错过截止</th></tr>'''+''.join(rows)+'''</table></div><p>周期处理时间包含结果接收、解析、核对及物理步进；调度迟到另外保存。复验只重放已保存的事件，不重复宣称一次并发测量。</p></section><section><a href="report.json">完整报告</a> · <a href="protocol.md">固定协议</a></section></main><script>document.getElementById('trial').addEventListener('change',function(){document.querySelectorAll('.trial').forEach(e=>e.style.display=e.dataset.index===this.value?'block':'none')})</script></html>'''


def run(output, verify):
    fixed = sources()
    if verify:
        count = check_manifest(output)
        saved = json.loads((output/'report.json').read_text(encoding='utf-8'))
        if saved['sources'] != fixed:
            raise ValueError('source binding changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'protocol.md').write_bytes((ROOT/'docs/ASYNC_STOP_PROTOCOL.md').read_bytes())
    trials = []
    for spec in TRIALS:
        folder = output/spec[0]
        if not verify: folder.mkdir()
        raw = verify_trial(spec, folder) if verify else run_trial(spec, folder)
        trial = summarize(spec, folder, raw)
        trials.append(trial)
        print(json.dumps({k:trial[k] for k in ('name', 'receive_wall_s', 'receive_step', 'matching_ticks',
              'first_failure', 'max_work_s', 'missed_deadlines', 'max_position_error_m')},ensure_ascii=False), flush=True)
    if sources() != fixed:
        raise ValueError('source changed during experiment')
    summary = dict(trials=len(trials), actual_steps=sum(t['actual_steps'] for t in trials),
        prediction_steps=sum(t['prediction_steps'] for t in trials), prefix_steps=sum(t['prefix_steps'] for t in trials),
        monitor_ticks=len(trials)*401, matched_trials=sum(t['matching_ticks'] > 0 for t in trials),
        actual_holds_confirmed=sum(t['actual_hold']['actual_hold_window']['final_window_confirmed'] for t in trials),
        missed_deadlines=sum(t['missed_deadlines'] for t in trials),
        source_states_advance_while_result_pending=all(t['receive_step'] is not None and t['receive_step'] > 0 for t in trials))
    report = dict(sources=fixed, parent_sha256=sha(PARENT/'report.json'), trials=trials, summary=summary,
        original_run='wall-clock-paced feedback with one independent worker process per trial',
        verification='deterministic prediction, saved-event feedback and independent motor replay; no new concurrent timing',
        hard_realtime_certified=False, execution_authorized=False, new_model_calls=0, real_flights=0)
    if verify:
        if report != saved or page(report) != (output/'demo.html').read_text(encoding='utf-8'):
            raise ValueError('report/page differs')
        if check_manifest(output) != count or (output/'protocol.md').read_bytes() != (ROOT/'docs/ASYNC_STOP_PROTOCOL.md').read_bytes():
            raise ValueError('archive changed')
    else:
        dump(output/'report.json', report)
        (output/'demo.html').write_text(page(report), encoding='utf-8')
        dump(output/'manifest.json', {p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
        count = check_manifest(output)
    return dict(verified=verify, files=count, sources=len(fixed), **summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve(), args.verify), ensure_ascii=False), flush=True)
