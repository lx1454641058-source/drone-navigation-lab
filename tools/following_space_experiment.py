"""来源：本项目原创。持续物理停止、异步空间查询及阶段时间证据。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import gc
import gzip
import html
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import threading
from time import perf_counter, sleep, thread_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drone_nav.async_stop import (make_request, request_record, StopStepper, checkpoint,
    predict_index, summarize_trace, ForecastMonitor)
from drone_nav.following_space import prepare_following_space, consume_following_space
from drone_nav.forecast_space import SimulationReference
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.stop_forecast import FrozenForecast, StopRecipe, CLOCK
from drone_nav.pinhole import Intrinsics, Pose
from drone_nav.raycast import World, Surface, Box, render
from drone_nav.realvision import sha, dump
from tools.async_stop_experiment import ready, predict_job, prepare, fault_command, write_gzip, independent_bounds
from tools.forecast_space_experiment import build_inspector, audit_geometry
from tools.braking_measurement_experiment import canonical, independent_metrics
from tools.stopping_policy_experiment import independent_hold
from tools.object_probe import check_manifest

PARENT = ROOT/'work/forecast-space-03'
OWN = ('drone_nav/following_space.py', 'tools/following_space_experiment.py',
       'tests/test_following_space.py', 'docs/FOLLOWING_SPACE_PROTOCOL.md')
CASES = (('far', '远墙：连续查询'), ('near', '近墙：保留表面限制'),
         ('missing', '缺测：保留未知'), ('late-space', '空间结果延迟'),
         ('late-thrust', '执行推力扰动'), ('control-stall', '控制循环卡顿'))


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def sources():
    check_manifest(PARENT)
    previous = read(PARENT/'report.json')['sources']
    for key, value in previous.items():
        if sha(ROOT/key) != value:
            raise ValueError('frozen source changed: '+key)
    return {**previous, **{key: sha(ROOT/key) for key in OWN}}


class StageClock:
    def __init__(self, started):
        self.started = started
        self.main_thread = threading.get_ident()
        self.phase = 'outside_cycle'
        self.gc_events = []
        self.gc_started = {}
        self.stages = {}

    def callback(self, phase, info):
        tid = threading.get_ident()
        key = (tid, info['generation'])
        if phase == 'start':
            self.gc_started[key] = (perf_counter()-self.started, thread_time(), self.phase if tid == self.main_thread else 'other_thread')
        elif key in self.gc_started:
            begin, cpu, stage = self.gc_started.pop(key)
            self.gc_events.append(dict(thread_id=tid, main_thread=tid == self.main_thread,
                generation=info['generation'], stage=stage, begin_s=begin,
                end_s=perf_counter()-self.started, thread_cpu_s=thread_time()-cpu,
                collected=info['collected'], uncollectable=info['uncollectable']))

    def begin(self, name):
        self.phase = name
        self.wall = perf_counter()
        self.cpu = thread_time()

    def end(self):
        self.stages[self.phase] = dict(wall_s=perf_counter()-self.wall, thread_cpu_s=thread_time()-self.cpu)
        self.phase = 'between_stages'


def compute_space(request, packet, task):
    # A fresh snapshot monitor validates packet/state identity, not delivery time.
    # The main monitor owns the real prediction reception deadline.
    monitor = ForecastMonitor(request)
    snapshot = monitor.tick(task['current'], elapsed_s=0., packet=packet)
    if not snapshot['forecast_matches']:
        raise ValueError('submitted checkpoint is not on this prediction')
    entry = task['entry_position']
    pose = Pose((entry[0]-2., entry[1], entry[2]), (0.,-1.,0.), (0.,0.,-1.), (1.,0.,0.))
    k = Intrinsics(80,60,60.,60.,39.5,29.5)
    wall_x = entry[0]+(.40 if task['mode'] == 'near' else 3.)
    world = World((Surface('ground',0,(-10.,10.,-10.,10.)),
                   Box('probe-wall',3,(wall_x,-3.,1.),(wall_x+.02,3.,6.))))
    frame, _ = render(world, pose, k, seed=3817, tick=task['current']['index']//10)
    if task['mode'] == 'missing':
        frame = replace(frame, depth_z_m=tuple(None if (n//k.width)%3 == 0 and n%k.width%3 == 0 else z
                                             for n,z in enumerate(frame.depth_z_m)))
    inspector, reference, provenance = build_inspector(frame, capture_s=task['captured_state']['time_s'],
        frame_id=task['id'], model_sha=task['current']['model_sha256'])
    receipt = prepare_following_space(monitor, task['current'], inspector, reference,
                                     now_s=task['current']['time_s'], clock_id=CLOCK)
    query = FrozenForecast(**receipt.record()['query']).record()
    audit = audit_geometry(query, inspector)
    evidence = canonical(dict(task=task, frame=asdict(frame), world=asdict(world),
        provenance=provenance, audit=audit, reference=asdict(reference), receipt=asdict(receipt)))
    return receipt, reference, evidence


def space_job(request, packet, task, folder):
    started = perf_counter()
    receipt, reference, evidence = compute_space(request, packet, task)
    computed = perf_counter()-started
    folder.mkdir()
    write_gzip(folder/'evidence.json.gz', evidence)
    from PIL import Image
    frame = evidence['frame']
    Image.frombytes('RGB', (80,60), bytes(v for rgb in frame['rgb'] for v in rgb)).save(folder/'input.png')
    before_wait = perf_counter()-started
    sleep(task['injected_wait_s'])
    metadata = dict(pid=os.getpid(), compute_s=computed, before_wait_s=before_wait,
                    injected_wait_s=task['injected_wait_s'], worker_elapsed_s=perf_counter()-started)
    dump(folder/'worker.json', metadata)
    return receipt, reference, metadata


def run_trial(spec, folder):
    name, _ = spec
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as executor:
        worker = executor.submit(ready).result(timeout=20)
        with QuadrotorPhysics() as physics:
            parent_path, parent = prepare(physics, 'original')
            request = make_request(physics, StopRecipe('original', control_seconds=.02), 'following-'+name)
            record, recipe = request_record(request)
            stepper = StopStepper(record['entry'], recipe, record['prior_motors'])
            monitor = ForecastMonitor(request)
            states, commands, events, tasks = [physics.state()], [], [], []
            prediction = None
            prediction_received = False
            space_future = None
            in_flight = active = None
            previous_missed = False
            started = perf_counter()
            timing = StageClock(started)
            gc.callbacks.append(timing.callback)
            try:
                future = executor.submit(predict_job, request, folder, 'normal')
                for tick in range(401):
                    timing.phase = 'sleep'
                    scheduled = tick*.02
                    remaining = started+scheduled-perf_counter()
                    if remaining > 0:
                        sleep(remaining)
                    if name == 'control-stall' and tick == 40:
                        sleep(.06)
                    begin, cpu_begin = perf_counter()-started, thread_time()
                    timing.stages = {}
                    timing.begin('checkpoint')
                    current = checkpoint(physics, stepper)
                    timing.end()
                    timing.begin('receive_prediction')
                    incoming = None
                    error = False
                    if not prediction_received and future.done():
                        prediction_received = True
                        try:
                            prediction, prediction_metadata = future.result()
                            incoming = prediction
                        except Exception as exc:
                            prediction_metadata = dict(error=f'{type(exc).__name__}: {exc}')
                            error = True
                    timing.end()
                    timing.begin('monitor')
                    elapsed = perf_counter()-started
                    missed_input = previous_missed or begin-scheduled > .02
                    decision = monitor.tick(current, elapsed_s=elapsed, packet=incoming,
                        worker_error=error, deadline_missed=missed_input)
                    timing.end()
                    timing.begin('receive_space')
                    space_received = space_error = None
                    if space_future is not None and space_future.done():
                        space_received = in_flight['id']
                        try:
                            receipt, reference, metadata = space_future.result()
                            active = (in_flight, receipt, reference)
                        except Exception as exc:
                            space_error = f'{type(exc).__name__}: {exc}'
                            active = None
                        space_future = in_flight = None
                    timing.end()
                    timing.begin('consume_space')
                    consumed_at = perf_counter()-started
                    completion_deadline = missed_input or consumed_at > scheduled+.02
                    spatial = None
                    if active is not None:
                        task, receipt, reference = active
                        spatial = consume_following_space(receipt, monitor, current, reference,
                            now_s=current['time_s'], clock_id=CLOCK,
                            wall_age_s=consumed_at-task['capture_wall_s'], deadline_missed=completion_deadline)
                    timing.end()
                    timing.begin('submit_space')
                    submitted = None
                    if 25 <= tick <= 350 and (tick-25)%10 == 0 and monitor.reason == 'MATCHED' and space_future is None:
                        submitted = f'query-{tick:03d}'
                        task = dict(id=submitted, mode=name, current=current, entry_position=states[0]['position'],
                            captured_state=states[(tick-5)*10], capture_tick=tick-5,
                            capture_wall_s=events[tick-5]['begin_s'], submitted_wall_s=perf_counter()-started,
                            injected_wait_s=.60 if name == 'late-space' else .08)
                        tasks.append(task)
                        in_flight = task
                        space_future = executor.submit(space_job, request, prediction, task, folder/submitted)
                    timing.end()
                    timing.begin('feedback')
                    if tick < 400:
                        for _ in range(10):
                            index = stepper.index
                            command = fault_command(stepper.command(states[-1]), index, name)
                            physics.step(command['motors'])
                            commands.append(command)
                            states.append(physics.state())
                    timing.end()
                    timing.begin('record_event')
                    event = dict(tick=tick, scheduled_s=scheduled, begin_s=begin, monitor_elapsed_s=elapsed,
                        checkpoint=current, delivered=incoming is not None, worker_error=error,
                        deadline_missed_input=missed_input, decision=decision, space_received=space_received,
                        space_error=space_error, active_task=None if active is None else active[0]['id'],
                        space_consumed_at_s=consumed_at, completion_deadline=completion_deadline,
                        spatial=spatial, submitted=submitted)
                    events.append(event)
                    timing.end()
                    end = perf_counter()-started
                    previous_missed = end > scheduled+.02
                    event.update(end_s=end, work_s=end-begin, thread_cpu_s=thread_time()-cpu_begin,
                        completed_after_next_deadline=previous_missed, stages=timing.stages)
                if not prediction_received or space_future is not None:
                    raise RuntimeError('worker not consumed by fixed horizon; preserve outputs')
                actual = summarize_trace(states, commands, stepper)
            finally:
                gc.callbacks.remove(timing.callback)
    raw = canonical(dict(name=name, request=asdict(request), actual=actual, events=events, tasks=tasks,
        gc_events=timing.gc_events, parent_input=parent_path.relative_to(ROOT).as_posix(), parent_sha256=sha(parent_path),
        prefix_steps=parent['request_index'], main_pid=os.getpid(), ready_worker=worker,
        prediction_metadata=prediction_metadata, flight_authorized=False, hard_realtime_certified=False))
    if name != 'late-thrust':
        assert actual['states'] == parent['states'][parent['request_index']:]
        assert [c['motors'] for c in commands] == [c['motors'] for c in parent['commands'][parent['request_index']:]]
    independent_metrics(actual)
    independent_hold(actual)
    write_gzip(folder/'actual.json.gz', raw)
    return raw


def verify_trial(spec, folder):
    name, _ = spec
    raw = read(folder/'actual.json.gz')
    request = FrozenForecast(**raw['request'])
    packet = FrozenForecast(**read(folder/'packet.json'))
    rebuilt, prediction = predict_index(request)
    assert rebuilt == packet and prediction == read(folder/'prediction.json.gz')
    independent_bounds(packet, prediction)
    task_map = {task['id']: task for task in raw['tasks']}
    received = {}
    for task in raw['tasks']:
        receipt, reference, evidence = compute_space(request, packet, task)
        assert evidence == read(folder/task['id']/'evidence.json.gz')
        received[task['id']] = (receipt, reference)
        # Direct full-tail scan is independent of the suffix index builder.
        sealed = receipt.record()
        for proof in sealed['proofs']:
            tail = prediction['states'][proof['index']:]
            for axis in range(3):
                assert abs(proof['lower'][axis]-(min(s['position'][axis] for s in tail)-.41)) < 1e-12
                assert abs(proof['upper'][axis]-(max(s['position'][axis] for s in tail)+.41)) < 1e-12
    monitor = ForecastMonitor(request)
    with QuadrotorPhysics() as physics:
        path, parent = prepare(physics, 'original')
        assert sha(path) == raw['parent_sha256'] and path.relative_to(ROOT).as_posix() == raw['parent_input']
        assert make_request(physics, StopRecipe('original',control_seconds=.02), 'following-'+name) == request
        record, recipe = request_record(request)
        stepper = StopStepper(record['entry'], recipe, record['prior_motors'])
        states, commands = [physics.state()], []
        active = in_flight = None
        for event in raw['events']:
            current = checkpoint(physics, stepper)
            assert current == event['checkpoint']
            decision = monitor.tick(current, elapsed_s=event['monitor_elapsed_s'],
                packet=packet if event['delivered'] else None, worker_error=event['worker_error'],
                deadline_missed=event['deadline_missed_input'])
            assert decision == event['decision']
            if event['space_received'] is not None:
                assert event['space_received'] == in_flight
                active = None if event['space_error'] else in_flight
                in_flight = None
            assert active == event['active_task']
            spatial = None
            if active is not None:
                receipt, reference = received[active]
                spatial = consume_following_space(receipt, monitor, current, reference,
                    now_s=current['time_s'],clock_id=CLOCK,
                    wall_age_s=event['space_consumed_at_s']-task_map[active]['capture_wall_s'],
                    deadline_missed=event['completion_deadline'])
            assert spatial == event['spatial']
            tick = event['tick']
            should_submit = 25 <= tick <= 350 and (tick-25)%10 == 0 and monitor.reason == 'MATCHED' and in_flight is None
            assert (event['submitted'] is not None) == should_submit
            if should_submit:
                task = task_map[event['submitted']]
                assert task['current'] == current and task['captured_state'] == states[(tick-5)*10]
                assert task['capture_wall_s'] == raw['events'][tick-5]['begin_s']
                assert task['entry_position'] == states[0]['position'] and task['capture_tick'] == tick-5
                assert task['mode'] == name
                assert task['injected_wait_s'] == (.60 if name == 'late-space' else .08)
                in_flight = event['submitted']
            if tick < 400:
                for _ in range(10):
                    index = stepper.index
                    command = fault_command(stepper.command(states[-1]), index, name)
                    physics.step(command['motors'])
                    commands.append(command)
                    states.append(physics.state())
        assert summarize_trace(states,commands,stepper) == raw['actual']
    with QuadrotorPhysics() as physics:
        prepare(physics, 'original')
        assert physics.state() == raw['actual']['states'][0]
        for index, command in enumerate(raw['actual']['commands'],1):
            physics.step(command['motors'])
            assert physics.state() == raw['actual']['states'][index]
    independent_metrics(raw['actual'])
    independent_hold(raw['actual'])
    return raw


def summarize(spec, folder, raw):
    events = raw['events']
    assert len(events) == 401 and [e['tick'] for e in events] == list(range(401))
    assert raw['ready_worker']['pid'] != raw['main_pid']
    previous = False
    for event in events:
        assert event['scheduled_s'] <= event['begin_s'] <= event['monitor_elapsed_s'] <= event['space_consumed_at_s'] <= event['end_s']
        assert event['deadline_missed_input'] == (previous or event['begin_s']-event['scheduled_s'] > .02)
        assert event['completion_deadline'] == (event['deadline_missed_input'] or event['space_consumed_at_s'] > event['scheduled_s']+.02)
        previous = event['end_s'] > event['scheduled_s']+.02
        assert previous == event['completed_after_next_deadline']
        assert event['work_s'] == event['end_s']-event['begin_s']
        assert sum(s['wall_s'] for s in event['stages'].values()) <= event['work_s']+1e-9
    deliveries = []
    for event in events:
        if event['space_received'] is not None:
            task = next(t for t in raw['tasks'] if t['id'] == event['space_received'])
            metadata = read(folder/task['id']/'worker.json')
            assert metadata['pid'] == raw['ready_worker']['pid']
            deliveries.append(dict(task=task['id'],query_index=task['current']['index'],
                receive_index=event['checkpoint']['index'],
                task_elapsed_s=event['space_consumed_at_s']-task['submitted_wall_s'],
                wall_age_s=event['space_consumed_at_s']-task['capture_wall_s'],
                diagnostic_current=False if event['spatial'] is None else event['spatial']['diagnostic_current'],
                reasons=[] if event['spatial'] is None else event['spatial']['reasons'],worker=metadata))
    valid = [e for e in events if e['spatial'] is not None and e['spatial']['diagnostic_current']]
    overruns = []
    for event in events:
        if event['completed_after_next_deadline']:
            largest = max(event['stages'],key=lambda k:event['stages'][k]['wall_s'])
            overlap = [g for g in raw['gc_events'] if g['main_thread'] and g['begin_s'] <= event['end_s'] and g['end_s'] >= event['begin_s']]
            overruns.append(dict(tick=event['tick'],work_s=event['work_s'],thread_cpu_s=event['thread_cpu_s'],
                start_lateness_s=event['begin_s']-event['scheduled_s'],largest_stage=largest,
                largest_stage_timing=event['stages'][largest],gc_overlap=overlap))
    return dict(name=spec[0],title=spec[1],queries=len(raw['tasks']),deliveries=deliveries,
        delivered_current=sum(d['diagnostic_current'] for d in deliveries),current_cycles=len(valid),
        all_deliveries_advanced=all(d['receive_index']>d['query_index'] for d in deliveries) if deliveries else None,
        final_monitor_reason=events[-1]['decision']['reason'],max_work_s=max(e['work_s'] for e in events),
        max_thread_cpu_s=max(e['thread_cpu_s'] for e in events),overruns=overruns,
        gc_events=len(raw['gc_events']),flight_authorized=False,
        stages={key:dict(max_wall_s=max(e['stages'][key]['wall_s'] for e in events),
                         max_thread_cpu_s=max(e['stages'][key]['thread_cpu_s'] for e in events)) for key in events[0]['stages']},
        image=None if not raw['tasks'] else raw['tasks'][0]['id']+'/input.png')


def page(report):
    panels=[]
    for n,row in enumerate(report['trials']):
        deliveries=''.join(f'<tr><td>{d["task"]}</td><td>{d["query_index"]} → {d["receive_index"]}</td><td>{d["wall_age_s"]*1000:.1f}</td><td>{"是" if d["diagnostic_current"] else "否"}</td></tr>' for d in row['deliveries'])
        img='' if row['image'] is None else f'<img src="{row["name"]}/{row["image"]}" alt="本组首帧理想观察相机图像">'
        panels.append(f'<section data-case="{n}" style="display:{"block" if n==0 else "none"}"><h2>{row["title"]}</h2>{img}<p>提交 {row["queries"]} 次；交付时仍适用 {row["delivered_current"]} 次；适用周期 {row["current_cycles"]}。</p><p>全部交付发生在状态前进后：{row["all_deliveries_advanced"]}。飞行许可：无。</p><p>最大周期处理 {row["max_work_s"]*1000:.2f} ms；最大线程 CPU {row["max_thread_cpu_s"]*1000:.2f} ms；超期周期 {len(row["overruns"])}。</p><p>监测终态：{html.escape(row["final_monitor_reason"])}</p><table><thead><tr><th>查询</th><th>物理步号：提交 → 消费</th><th>观测壁钟年龄 ms</th><th>诊断适用</th></tr></thead><tbody>{deliveries}</tbody></table><details><summary>分段计时、失效原因与 GC 原始摘要</summary><pre>{html.escape(json.dumps(row,ensure_ascii=False,indent=2))}</pre></details></section>')
    options=''.join(f'<option value="{i}">{r["title"]}</option>' for i,r in enumerate(report['trials']))
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>运动中的停止空间检查</title><style>body{background:#eef3f5;color:#1b3345;font:16px/1.7 system-ui;margin:0;padding:24px}main{max-width:1100px;margin:auto}section{background:white;padding:24px;border-radius:12px;margin-top:20px}h1{font-size:30px}select{font:inherit;padding:8px;max-width:100%}img{width:240px;image-rendering:pixelated;float:right;margin:0 0 18px 24px}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #dfe6ea;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.note{color:#89501e}@media(max-width:650px){img{float:none;margin:0;width:100%}body{padding:12px}section{padding:16px}}</style><main><h1>空间检查回来时，无人机已经前进了</h1><p>异步预测与空间查询 · 持续停止反馈 · 同预测剩余范围包含检查 · 分别检查模型时间与真实等待时间</p><p class="note">理想相机和候选墙面的接口实验。保留缺测、表面、覆盖与模型误差限制，诊断仍适用不表示允许飞行；不是实际无人机演示。</p><label for="case">场景：</label><select id="case">'''+options+'''</select>'''+''.join(panels)+'''<section><h2>如何理解</h2><p>每次普通查询额外等待 80 ms，延迟组为 600 ms，等待计入年龄。状态继续推进时，只有同一预测仍匹配且剩余范围包含在原检查区域内，旧诊断才继续适用。</p><p>外层检查的表面计数不重新算为当前小区域内的障碍。空间未知不会变成空闲。控制器始终执行原停止反馈。</p><p>垃圾回收重叠或壁钟大于线程 CPU 都只是定位线索，不能单独归因。重放复用已记录时间，不是第二次并发性能测试。</p><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定协议</a></p></section></main><script>document.getElementById('case').addEventListener('change',function(){document.querySelectorAll('[data-case]').forEach(e=>e.style.display=e.dataset.case===this.value?'block':'none')})</script></html>'''


def run(output, verify):
    fixed=sources()
    if verify:
        count=check_manifest(output)
        saved=read(output/'report.json')
        assert saved['sources']==fixed
    else:
        output.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(ROOT/'docs/FOLLOWING_SPACE_PROTOCOL.md',output/'protocol.md')
    rows=[]
    for spec in CASES:
        folder=output/spec[0]
        if not verify: folder.mkdir()
        raw=verify_trial(spec,folder) if verify else run_trial(spec,folder)
        row=summarize(spec,folder,raw)
        rows.append(row)
        print(json.dumps({k:row[k] for k in ('name','queries','delivered_current','current_cycles','max_work_s','final_monitor_reason')},ensure_ascii=False),flush=True)
    report=dict(sources=fixed,parent_manifest_sha256=sha(PARENT/'manifest.json'),trials=rows,
        actual_stop_steps=24000,prediction_steps=24000,events=2406,
        queries=sum(r['queries'] for r in rows),current_deliveries=sum(r['delivered_current'] for r in rows),
        missed_deadlines=sum(len(r['overruns']) for r in rows),flight_authorized=False,new_neural_model_calls=0)
    assert sources()==fixed
    if verify:
        assert saved==report and (output/'demo.html').read_text(encoding='utf-8')==page(report)
        assert check_manifest(output)==count
    else:
        dump(output/'report.json',report)
        (output/'demo.html').write_text(page(report),encoding='utf-8')
        dump(output/'manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
        count=check_manifest(output)
    return dict(verified=verify,files=count,sources=len(fixed),queries=report['queries'],current_deliveries=report['current_deliveries'],missed_deadlines=report['missed_deadlines'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.output.resolve(),args.verify),ensure_ascii=False),flush=True)
