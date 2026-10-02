"""来源：本项目原创。无损紧凑记录的配对测量，以及消费结束后的截止复查。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gc
import gzip
import html
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter, sleep, thread_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drone_nav.async_stop import (make_request, request_record, StopStepper, checkpoint,
    predict_index, summarize_trace, ForecastMonitor)
from drone_nav.compact_stop_log import CompactStopLog
from drone_nav.finished_space import consume_finished_space
from tools.following_space_experiment import StageClock, compute_space, space_job
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.stop_forecast import FrozenForecast, StopRecipe, CLOCK
from drone_nav.realvision import sha, dump
from tools.async_stop_experiment import ready, predict_job, prepare, fault_command, write_gzip, independent_bounds
from tools.braking_measurement_experiment import canonical, independent_metrics
from tools.stopping_policy_experiment import independent_hold
from tools.object_probe import check_manifest

PARENT = ROOT/'work/following-space-01'
OWN = ('drone_nav/compact_stop_log.py', 'drone_nav/finished_space.py',
       'tools/compact_stop_experiment.py', 'tests/test_compact_stop.py', 'docs/COMPACT_STOP_PROTOCOL.md')
# ABBA order; each trial runs in a fresh process, with a separate prediction worker.
CASES = (('rich-a','原记录 A','rich',0.,'far'),
         ('compact-a','紧凑记录 A','compact',0.,'far'),
         ('compact-b','紧凑记录 B','compact',0.,'far'),
         ('rich-b','原记录 B','rich',0.,'far'),
         ('finish-late','消费期间延迟 60 ms','compact',.06,'far'),
         ('finish-expired','消费期间延迟 500 ms','compact',.50,'far'),
         ('compact-thrust','紧凑记录与推力扰动','compact',0.,'late-thrust'))


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


def run_trial(spec, folder):
    name, title, storage, finish_delay, scenario = spec
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as executor:
        worker = executor.submit(ready).result(timeout=20)
        with QuadrotorPhysics() as physics:
            parent_path, parent = prepare(physics, 'original')
            request = make_request(physics, StopRecipe('original', control_seconds=.02), 'compact-study-'+name)
            record, recipe = request_record(request)
            stepper = StopStepper(record['entry'], recipe, record['prior_motors'])
            monitor = ForecastMonitor(request)
            state = physics.state()
            compact = CompactStopLog(state) if storage == 'compact' else None
            states, commands, events, tasks = [state], [], [], []
            finish_injected = False
            scalar_log = None
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
                    delay_this_cycle = 0.
                    if active is not None:
                        task, receipt, reference = active
                        if finish_delay and not finish_injected and monitor.reason == 'MATCHED':
                            delay_this_cycle = finish_delay
                            finish_injected = True
                        def completion_checkpoint():
                            if delay_this_cycle:
                                sleep(delay_this_cycle)
                            return checkpoint(physics, stepper)
                        spatial = consume_finished_space(receipt, monitor, current, reference,
                            capture_wall_s=task['capture_wall_s'], cycle_deadline_wall_s=scheduled+.02,
                            clock_id=CLOCK, checkpoint_reader=completion_checkpoint,
                            clock_reader=lambda:perf_counter()-started, deadline_missed=completion_deadline)
                    timing.end()
                    timing.begin('submit_space')
                    submitted = None
                    if 25 <= tick <= 350 and (tick-25)%10 == 0 and monitor.reason == 'MATCHED' and space_future is None:
                        submitted = f'query-{tick:03d}'
                        task = dict(id=submitted, mode=scenario, current=current, entry_position=record['entry']['position'],
                            captured_state=compact.state((tick-5)*10) if compact is not None else states[(tick-5)*10], capture_tick=tick-5,
                            capture_wall_s=events[tick-5]['begin_s'], submitted_wall_s=perf_counter()-started,
                            injected_wait_s=.08)
                        tasks.append(task)
                        in_flight = task
                        space_future = executor.submit(space_job, request, prediction, task, folder/submitted)
                    timing.end()
                    timing.begin('feedback')
                    if tick < 400:
                        for _ in range(10):
                            index = stepper.index
                            command = stepper.command(state)
                            if scenario == 'late-thrust' and 500 <= index < 600:
                                command['motors'] = [v*.85 for v in command['motors']]
                            if compact is None:
                                command = fault_command(command,index,'normal')
                            physics.step(command['motors'])
                            state = physics.state()
                            if compact is not None:
                                compact.append(command['motors'],state)
                            else:
                                commands.append(command)
                                states.append(state)
                    timing.end()
                    timing.begin('record_event')
                    event = dict(tick=tick, scheduled_s=scheduled, begin_s=begin, monitor_elapsed_s=elapsed,
                        checkpoint=current, delivered=incoming is not None, worker_error=error,
                        deadline_missed_input=missed_input, decision=decision, space_received=space_received,
                        space_error=space_error, active_task=None if active is None else active[0]['id'],
                        space_consumed_at_s=consumed_at, completion_deadline=completion_deadline,
                        spatial=spatial, submitted=submitted, finish_delay_s=delay_this_cycle)
                    events.append(event)
                    timing.end()
                    end = perf_counter()-started
                    previous_missed = end > scheduled+.02
                    event.update(end_s=end, work_s=end-begin, thread_cpu_s=thread_time()-cpu_begin,
                        completed_after_next_deadline=previous_missed, stages=timing.stages)
                if not prediction_received or space_future is not None:
                    raise RuntimeError('worker not consumed by fixed horizon; preserve outputs')
                timing.phase = 'post_loop_expansion'
                if compact is not None:
                    scalar_log = dict(schema='actual-stop-scalars-v1', state_width=14, motor_width=4,
                        steps=compact.count, states=list(compact.states), motors=list(compact.controls),
                        native_buffer_bytes=(len(compact.states)+len(compact.controls))*compact.states.itemsize)
                    states, commands, rebuilt = compact.expand(record,recipe,
                        transform_command=lambda c,i:fault_command(c,i,scenario))
                    if canonical(rebuilt.context()) != canonical(stepper.context()):
                        raise ValueError('compact annotation reconstruction changed final controller context')
                actual = summarize_trace(states, commands, stepper)
            finally:
                gc.callbacks.remove(timing.callback)
    raw = canonical(dict(name=name, storage=storage, scenario=scenario, finish_delay_s=finish_delay,
        command_annotations='reconstructed_after_loop_and_checked_against_actual_motors' if compact is not None else 'recorded_during_loop',
        gc_enabled=gc.isenabled(), request=asdict(request), actual=actual, events=events, tasks=tasks,
        gc_events=timing.gc_events, parent_input=parent_path.relative_to(ROOT).as_posix(), parent_sha256=sha(parent_path),
        prefix_steps=parent['request_index'], main_pid=os.getpid(), ready_worker=worker,
        prediction_metadata=prediction_metadata, flight_authorized=False, hard_realtime_certified=False))
    if scenario != 'late-thrust':
        assert actual['states'] == parent['states'][parent['request_index']:]
        assert [c['motors'] for c in commands] == [c['motors'] for c in parent['commands'][parent['request_index']:]]
    independent_metrics(actual)
    independent_hold(actual)
    if scalar_log is not None:
        write_gzip(folder/'scalar-log.json.gz',scalar_log)
    write_gzip(folder/'actual.json.gz', raw)
    return raw


def verify_trial(spec, folder):
    name, title, storage, finish_delay, scenario = spec
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
        assert make_request(physics, StopRecipe('original',control_seconds=.02), 'compact-study-'+name) == request
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
                saved_clock=iter((event['spatial']['started_wall_s'],event['spatial']['finished_wall_s']))
                spatial = consume_finished_space(receipt,monitor,current,reference,
                    capture_wall_s=task_map[active]['capture_wall_s'],cycle_deadline_wall_s=event['scheduled_s']+.02,
                    clock_id=CLOCK,checkpoint_reader=lambda:checkpoint(physics,stepper),
                    clock_reader=lambda:next(saved_clock),deadline_missed=event['completion_deadline'])
            assert spatial == event['spatial']
            tick = event['tick']
            should_submit = 25 <= tick <= 350 and (tick-25)%10 == 0 and monitor.reason == 'MATCHED' and in_flight is None
            assert (event['submitted'] is not None) == should_submit
            if should_submit:
                task = task_map[event['submitted']]
                assert task['current'] == current and task['captured_state'] == states[(tick-5)*10]
                assert task['capture_wall_s'] == raw['events'][tick-5]['begin_s']
                assert task['entry_position'] == states[0]['position'] and task['capture_tick'] == tick-5
                assert task['mode'] == scenario
                assert task['injected_wait_s'] == .08
                in_flight = event['submitted']
            if tick < 400:
                for _ in range(10):
                    index = stepper.index
                    command = fault_command(stepper.command(states[-1]), index, scenario)
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
    if storage == 'compact':
        scalars=read(folder/'scalar-log.json.gz')
        assert scalars['steps']==4000 and scalars['native_buffer_bytes']==576112
        for index,state in enumerate(raw['actual']['states']):
            assert scalars['states'][index*14:(index+1)*14]==[state['time_s'],*state['position'],*state['quaternion'],*state['velocity'],*state['angular_velocity']]
        assert scalars['motors']==[v for command in raw['actual']['commands'] for v in command['motors']]
    injected=[e for e in raw['events'] if e['finish_delay_s']]
    assert all(e['finish_delay_s']==finish_delay for e in injected)
    if finish_delay:
        assert len(injected)==1
        assert injected[0]['spatial']['finished_wall_s']-injected[0]['spatial']['started_wall_s']>=finish_delay
    else:
        assert not injected
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
                wall_age_s=None if event['spatial'] is None else event['spatial']['final_wall_age_s'],
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
    from statistics import median
    injected=[e for e in events if e['finish_delay_s']]
    return dict(name=spec[0],title=spec[1],storage=spec[2],finish_delay_s=spec[3],
        feedback_median_s=median(e['stages']['feedback']['wall_s'] for e in events[:-1]),
        feedback_total_s=sum(e['stages']['feedback']['wall_s'] for e in events[:-1]),
        final_check_rejections=sum(e['spatial'] is not None and e['spatial']['provisional']['diagnostic_current'] and not e['spatial']['diagnostic_current'] for e in events),
        injected=[dict(tick=e['tick'],spatial=e['spatial']) for e in injected],
        main_gc_in_cycle=[g for g in raw['gc_events'] if g['main_thread'] and g['stage']!='post_loop_expansion'],
        gc_enabled=raw['gc_enabled'],queries=len(raw['tasks']),deliveries=deliveries,
        delivered_current=sum(d['diagnostic_current'] for d in deliveries),current_cycles=len(valid),
        all_deliveries_advanced=all(d['receive_index']>d['query_index'] for d in deliveries) if deliveries else None,
        final_monitor_reason=events[-1]['decision']['reason'],max_work_s=max(e['work_s'] for e in events),
        max_thread_cpu_s=max(e['thread_cpu_s'] for e in events),overruns=overruns,
        gc_events=len(raw['gc_events']),flight_authorized=False,
        stages={key:dict(max_wall_s=max(e['stages'][key]['wall_s'] for e in events),
                         max_thread_cpu_s=max(e['stages'][key]['thread_cpu_s'] for e in events)) for key in events[0]['stages']},
        image=None if not raw['tasks'] else raw['tasks'][0]['id']+'/input.png')


def page(report):
    rows=[]
    panels=[]
    for row in report['trials']:
        rows.append(f'<tr><td>{row["title"]}</td><td>{row["feedback_median_s"]*1000:.3f}</td><td>{row["max_work_s"]*1000:.2f}</td><td>{len(row["overruns"])}</td><td>{row["final_check_rejections"]}</td></tr>')
        panels.append('<details><summary>'+html.escape(row['title'])+'：记录与最终复查</summary><pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre></details>')
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>紧凑控制记录与最终时效复查</title><style>body{background:#eef3f5;color:#20394b;font:16px/1.7 system-ui;margin:0;padding:24px}main{max-width:1100px;margin:auto}section{background:white;border-radius:12px;padding:24px;margin:20px 0}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #dce4e8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}details{margin:16px 0}.note{color:#8b5422}</style><main><h1>记录更轻，结果结束时再查一次</h1><p>原记录 A → 紧凑 A → 紧凑 B → 原记录 B；每次独立进程，保持原控制器与垃圾回收开启。</p><section><p>紧凑路径逐步记录实际状态和电机指令，详细说明在循环外重建并逐项核对。四次正常对照的全部实际状态和控制指令一致。</p><p class="note">本轮是电脑仿真停止反馈实验。未知空间、模型误差和视野限制仍保留，无导航飞行许可。</p><table><thead><tr><th>实验</th><th>反馈中位 ms</th><th>最大周期 ms</th><th>超期周期</th><th>开始有效、结束拒绝</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></section><section><h2>时间口径</h2><p>60 / 500 ms 延迟故意发生在消费开始与最终检查点之间，计入超期。最终复查会拒绝处理期间过期的结果，不沿用开始时刻。</p><p>线程 CPU 读数较粗，原值仅供参考。每种记录两次不证明稳定实时能力；结果复验是保存时刻的重放，不是又一次并发测速。</p>'+''.join(panels)+'<p><a href="report.json">完整报告</a> · <a href="protocol.md">固定协议</a></p></section></main></html>'


def isolated_trial(spec,folder):
    run_trial(spec,folder)
    return dict(pid=os.getpid(),name=spec[0])


def run(output, verify):
    fixed=sources()
    if verify:
        count=check_manifest(output)
        saved=read(output/'report.json')
        assert saved['sources']==fixed
    else:
        output.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(ROOT/'docs/COMPACT_STOP_PROTOCOL.md',output/'protocol.md')
    rows=[]
    for spec in CASES:
        folder=output/spec[0]
        if not verify: folder.mkdir()
        if verify:
            raw=verify_trial(spec,folder)
        else:
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as process:
                process.submit(isolated_trial,spec,folder).result(timeout=60)
            raw=read(folder/'actual.json.gz')
        row=summarize(spec,folder,raw)
        rows.append(row)
        print(json.dumps({k:row[k] for k in ('name','queries','delivered_current','current_cycles','max_work_s','final_monitor_reason')},ensure_ascii=False),flush=True)
    report=dict(sources=fixed,parent_manifest_sha256=sha(PARENT/'manifest.json'),trials=rows,
        actual_stop_steps=28000,prediction_steps=28000,events=2807,
        queries=sum(r['queries'] for r in rows),current_deliveries=sum(r['delivered_current'] for r in rows),
        missed_deadlines=sum(len(r['overruns']) for r in rows),flight_authorized=False,new_neural_model_calls=0)
    nominal=[read(output/spec[0]/'actual.json.gz')['actual'] for spec in CASES[:4]]
    if any(value!=nominal[0] for value in nominal[1:]):
        raise ValueError('paired logging variants changed actual feedback states or commands')
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
