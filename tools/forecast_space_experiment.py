"""来源：本项目原创。保存的物理预测与理想深度观测组合；不执行新飞行。"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
import html
from itertools import product
import json
from math import ceil, floor, sqrt
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drone_nav.async_stop import ForecastMonitor
from drone_nav.forecast_space import SimulationReference, query_stop_space, finish_stop_space
from drone_nav.stop_forecast import FrozenForecast, digest, CLOCK, WORLD
from drone_nav.depth_volume import DepthVolumeInspector
from drone_nav.detection_bridge import DetectionPacket, SpatialContext, project_detections
from drone_nav.frame_transform import RigidFrameTransform, CameraPoseStamp, transform_delivery, finish_transform
from drone_nav.framed_surface_map import transform_fingerprint
from drone_nav.timed_inbox import TimedObservationInbox
from drone_nav.packed_rgbd import PackedFrame, PackedObservation, load_packed
from drone_nav.pinhole import Intrinsics, Pose
from drone_nav.physical_vehicle import rotate_vector
from drone_nav.raycast import World, Surface, Box, render
from drone_nav.realvision import sha, dump
from drone_nav.tum_rgbd import INTRINSICS as TUM_K, CLOCK as TUM_CLOCK
from tools.framed_surface_map_experiment import reference as tum_reference
from tools.object_probe import check_manifest
from tools.stopping_policy_experiment import read
from tools.braking_measurement_experiment import canonical

PARENT = ROOT/'work/async-stop-01'
DEPTH_PARENT = ROOT/'work/depth-volume-01'
DATASET = ROOT/'work/tum-rgbd-bag-input-01'
CONVERTED = ROOT/'work/frame-transform-01'
OWN = ('drone_nav/forecast_space.py', 'tools/forecast_space_experiment.py',
       'tests/test_forecast_space.py', 'docs/FORECAST_SPACE_PROTOCOL.md',
       'drone_nav/raycast.py', 'drone_nav/physical_vehicle.py')
CASES = (
    ('early-far', '早期：远处采样', 'original-normal', 190, 'far'),
    ('early-near', '早期：范围内表面', 'original-normal', 190, 'near'),
    ('early-missing', '早期：深度缺测', 'original-normal', 190, 'missing'),
    ('onboard-blind', '机载前视：起始机体盲区', 'original-normal', 190, 'onboard'),
    ('late-far', '晚期：期限满足但覆盖未知', 'original-normal', 3800, 'far'),
    ('late-near', '晚期：期限满足但有表面', 'original-normal', 3800, 'near'),
    ('diverged', '预测已经失配', 'late-thrust', 510, 'far'),
    ('query-expired', '查询完成时观测过期', 'original-normal', 190, 'far'),
    ('state-advanced', '查询期间状态前进', 'original-normal', 190, 'far'),
    ('wrong-reference', '参考摘要不同', 'original-normal', 190, 'far'),
    ('wrong-model', '机体模型不同', 'original-normal', 190, 'far'),
    ('tum-unregistered', '真实 TUM：尚未对齐', 'original-normal', 190, 'tum'),
)


def sources():
    values = {}
    for folder in (PARENT, DEPTH_PARENT):
        check_manifest(folder)
        parent = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        for name, value in parent['sources'].items():
            if (name in values and values[name] != value) or sha(ROOT/name) != value:
                raise ValueError('frozen source differs: '+name)
            values[name] = value
    return {**values, **{name:sha(ROOT/name) for name in OWN}}


def build_inspector(frame, *, capture_s, frame_id, model_sha, camera_id='sim-stop-observer'):
    """Actual existing projection/inbox/transform pipeline, with explicit empty fixture detections."""
    k, pose = frame.intrinsics, frame.pose
    frame_bytes = bytes(v for rgb in frame.rgb for v in rgb)
    packed = PackedFrame(k, pose, frame_bytes, tuple(frame.depth_z_m))
    depth_source = 'ideal synthetic optical-Z raycast; no real camera'
    observation = PackedObservation(packed, frame_id, capture_s, capture_s, capture_s, depth_source)
    packet = DetectionPacket(frame_id, camera_id, k.width, k.height, (),
        'explicit empty fixture; no detector invoked', capture_s, capture_s+.02, CLOCK)
    context = SpatialContext(frame_id, camera_id, CLOCK, capture_s, capture_s, k, pose, packed.depth_z_m,
        True, 'camera_optical_axis_z_m', depth_source, 'known synthetic camera pose', 'fixed synthetic pinhole')
    available = capture_s+.04
    projection = project_detections(packet, context, now_s=available, now_clock_id=CLOCK)
    result = dict(projection=projection, reason='NO_DETECTION_REQUIRES_GEOMETRY',
                  flight_authorized=False, navigation_frame_alignment_available=False)
    inbox = TimedObservationInbox(camera_id=camera_id, clock_id=CLOCK, world_frame='east_north_up_m')
    submitted = inbox.submit(result, now_s=available, clock_id=CLOCK)
    if not submitted['accepted']: raise ValueError('synthetic observation rejected')
    delivered = inbox.consume(now_s=available, clock_id=CLOCK, target_world='east_north_up_m')
    transform = RigidFrameTransform('east_north_up_m', WORLD,
        ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)), (0.,0.,0.), 'declared-synthetic-world-identity',
        digest(dict(model=model_sha, coordinates='fixed known synthetic identity, not real calibration')))
    stamp = CameraPoseStamp(frame_id, camera_id, CLOCK, capture_s, pose, k)
    mapped = transform_delivery(delivered, transform, stamp, now_s=available, clock_id=CLOCK, target_frame=WORLD)
    finished = finish_transform(mapped, now_s=available, clock_id=CLOCK)
    inspector = DepthVolumeInspector(observation, finished, transform,
        camera_id=camera_id, clock_id=CLOCK, expected_intrinsics=k)
    reference = SimulationReference(WORLD, CLOCK, camera_id, model_sha, transform_fingerprint(transform))
    provenance = dict(packet=asdict(packet), submitted=submitted, delivered=delivered, finished=finished,
                      synthetic_only=True, actual_detector_called=False)
    return inspector, reference, canonical(provenance)


def replay_until(name, index):
    raw = read(PARENT/name/'actual.json.gz')
    request = FrozenForecast(**raw['request'])
    packet = FrozenForecast(**json.loads((PARENT/name/'packet.json').read_text(encoding='utf-8')))
    monitor = ForecastMonitor(request)
    for event in raw['events'][:index//10+1]:
        decision = monitor.tick(event['checkpoint'], elapsed_s=event['monitor_elapsed_s'],
            packet=packet if event['delivered'] else None, worker_error=event['worker_error'],
            deadline_missed=event['deadline_missed_input'])
        if decision != event['decision']: raise ValueError('parent monitor replay differs')
    return raw, monitor, raw['events'][index//10]['checkpoint']


def synthetic_frame(raw, index, mode):
    entry = raw['actual']['states'][0]['position']
    observed = raw['actual']['states'][index-50]
    if mode == 'onboard':
        xyz, q = tuple(observed['position']), observed['quaternion']
        pose = Pose(xyz, rotate_vector((0.,-1.,0.),q), rotate_vector((0.,0.,-1.),q), rotate_vector((1.,0.,0.),q))
    else:
        pose = Pose((entry[0]-2., entry[1], entry[2]), (0.,-1.,0.), (0.,0.,-1.), (1.,0.,0.))
    k = Intrinsics(80,60,60.,60.,39.5,29.5)
    wall_x = entry[0]+(.40 if mode == 'near' else 3.)
    world = World((Surface('ground',0,(-10.,10.,-10.,10.)),
                   Box('probe-wall',3,(wall_x,-3.,1.),(wall_x+.02,3.,6.))))
    frame, _ = render(world, pose, k, seed=3817, tick=index//10)
    if mode == 'missing':
        frame = replace(frame, depth_z_m=tuple(None if (i//k.width)%3 == 0 and i%k.width%3 == 0 else v
                                              for i,v in enumerate(frame.depth_z_m)))
    return frame, observed['time_s'], asdict(world)


def real_inspector():
    for path in (DATASET, CONVERTED): check_manifest(path)
    selected = json.loads((DATASET/'selection.json').read_text(encoding='utf-8'))
    converted = json.loads((CONVERTED/'report.json').read_text(encoding='utf-8'))
    observation = load_packed(DATASET, selected['frames'][0], selected['origin_s'])
    transform = tum_reference(converted)
    inspector = DepthVolumeInspector(observation, converted['frames'][0]['finished'], transform,
        camera_id='tum-freiburg3-rgb', clock_id=TUM_CLOCK, expected_intrinsics=TUM_K)
    return inspector, DATASET/selected['frames'][0]['rgb'][1], dict(
        dataset_manifest_sha256=sha(DATASET/'manifest.json'), converted_report_sha256=sha(CONVERTED/'report.json'),
        source='saved TUM registered RGB-D, no new inference or invented navigation alignment')


def audit_geometry(query, inspector):
    geometry, volume = query['geometry'], query['volume']
    if geometry is None: return dict(checked_pixels=0, skipped=True)
    if geometry['status'] == 'CONTEXT_REJECTED':
        assert query['checked_at_s'] > inspector.valid_until_s+1e-9
        return dict(checked_pixels=0, expired=True)
    pose,k = inspector.pose,inspector.intrinsics
    def camera(p):
        offset = [p[j]-pose.position[j] for j in range(3)]
        return [sum(offset[j]*axis[j] for j in range(3)) for axis in (pose.right,pose.down,pose.forward)]
    corners = [camera(p) for p in product(*zip(volume['lower'],volume['upper']))]
    if any(p[2] <= 0 for p in corners):
        assert geometry['reasons'] == ['VOLUME_NOT_FULLY_IN_FRONT']
        return dict(checked_pixels=0, behind_camera=True)
    projected = [(k.fx*p[0]/p[2]+k.cx, k.fy*p[1]/p[2]+k.cy, p[2]) for p in corners]
    window = [ceil(min(p[0] for p in projected)-.5-1e-9), ceil(min(p[1] for p in projected)-.5-1e-9),
              floor(max(p[0] for p in projected)+.5+1e-9), floor(max(p[1] for p in projected)+.5+1e-9)]
    assert window == geometry['window']
    x0,y0,x1,y1 = window
    if x0 < 0 or y0 < 0 or x1 >= k.width or y1 >= k.height:
        assert geometry['reasons'] == ['INCOMPLETE_VOLUME_VIEW']
        return dict(checked_pixels=0, image_boundary=True)
    far = max(p[2] for p in corners)
    counters = dict(checked_pixels=0,missing_pixels=0,out_of_range_pixels=0,foreground_pixels=0,inside_surface_pixels=0)
    for v in range(y0,y1+1):
        for u in range(x0,x1+1):
            counters['checked_pixels'] += 1
            z = inspector.depths[v*k.width+u]
            if z is None:
                counters['missing_pixels'] += 1; continue
            x,y = (u-k.cx)/k.fx, (v-k.cy)/k.fy
            if z*sqrt(x*x+y*y+1) > geometry['max_range_m']:
                counters['out_of_range_pixels'] += 1; continue
            counters['foreground_pixels'] += z-geometry['depth_error_m'] <= far+1e-9
            p = [pose.position[j]+z*(x*pose.right[j]+y*pose.down[j]+pose.forward[j]) for j in range(3)]
            counters['inside_surface_pixels'] += all(a-1e-9<=c<=b+1e-9 for a,c,b in zip(volume['lower'],p,volume['upper']))
    assert all(geometry[k] == v for k,v in counters.items()), (geometry,counters)
    return dict(counters, independently_projected=True)


def timing_review():
    rows = []
    for path in sorted(PARENT.glob('*/actual.json.gz')):
        for event in read(path)['events']:
            if event['completed_after_next_deadline']:
                rows.append(dict(case=path.parent.name,tick=event['tick'],work_s=event['work_s'],
                    lateness_s=event['start_lateness_s'],result_delivered=event['delivered'],
                    category='injected_stall_or_catchup' if path.parent.name=='control-stall'
                    else 'long_cycle' if event['work_s']>.02 else 'catchup_after_long_cycle'))
    assert len(rows)==10 and not any(r['result_delivered'] for r in rows)
    return dict(events=rows, categories=dict(Counter(r['category'] for r in rows)),
                specific_root_cause='unresolved: no stage CPU/GC profiling in original evidence',new_concurrent_measurement=False)


def run_case(spec, folder, verify):
    name,title,parent,index,mode = spec
    raw,monitor,current = replay_until(parent,index)
    now = current['time_s']
    if mode == 'tum':
        inspector,image_path,provenance = real_inspector()
        # Expected simulation reference is not rewritten to make the real input fit.
        reference = SimulationReference(WORLD,CLOCK,'sim-stop-observer',monitor.record['binding']['model_sha256'],'a'*64)
        frame_record = dict(source=provenance,image_sha256=sha(image_path))
        if not verify: shutil.copyfile(image_path,folder/'input.png')
    else:
        frame,capture,world = synthetic_frame(raw,index,mode)
        inspector,reference,provenance = build_inspector(frame,capture_s=capture,frame_id=name,
            model_sha=monitor.record['binding']['model_sha256'],camera_id='sim-onboard-forward' if mode=='onboard' else 'sim-stop-observer')
        frame_record = dict(frame=asdict(frame),world=world,provenance=provenance)
        if not verify:
            from PIL import Image
            Image.frombytes('RGB',(frame.intrinsics.width,frame.intrinsics.height),bytes(v for p in frame.rgb for v in p)).save(folder/'input.png')
    if name == 'wrong-reference': reference = replace(reference,reference_fingerprint='b'*64)
    if name == 'wrong-model': reference = replace(reference,model_sha256='c'*64)
    started = perf_counter()
    pending = query_stop_space(monitor,current,inspector,reference,now_s=now,clock_id=CLOCK)
    fresh_duration = perf_counter()-started
    timing = json.loads((folder/'timing.json').read_text(encoding='utf-8')) if verify else dict(
        query_seconds=fresh_duration, scope='offline query only; excludes render, pipeline setup and completion check')
    current_after = current
    if name == 'state-advanced':
        event = raw['events'][index//10+1]
        monitor.tick(event['checkpoint'],elapsed_s=event['monitor_elapsed_s'],
            worker_error=event['worker_error'],deadline_missed=event['deadline_missed_input'])
        current_after = event['checkpoint']
    injected_delay = .41 if name == 'query-expired' else 0.
    completion = max(now+timing['query_seconds']+injected_delay,current_after['time_s'])
    final = finish_stop_space(pending,monitor,current_after,inspector,now_s=completion,clock_id=CLOCK)
    query = pending.record()
    audit = audit_geometry(query,inspector)
    if query['volume'] is not None:
        prediction = read(PARENT/parent/'prediction.json.gz')
        tail = prediction['states'][index:]
        for j in range(3):
            assert abs(query['volume']['lower'][j]-(min(s['position'][j] for s in tail)-.41)) < 1e-12
            assert abs(query['volume']['upper'][j]-(max(s['position'][j] for s in tail)+.41)) < 1e-12
        assert query['horizon_end_s'] == prediction['states'][-1]['time_s']
    evidence = canonical(dict(spec=spec,parent_input_sha256=sha(PARENT/parent/'actual.json.gz'),
        frame=frame_record,query_sha256=pending.sha256,final=final,audit=audit,
        injected_completion_delay_s=injected_delay,actual_new_physics_steps=0,new_model_calls=0))
    if verify:
        if evidence != read(folder/'evidence.json.gz'): raise ValueError('combined replay differs: '+name)
    else:
        with (folder/'evidence.json.gz').open('xb') as stream:
            stream.write(gzip.compress(json.dumps(evidence,separators=(',',':'),allow_nan=False).encode(),mtime=0))
        dump(folder/'timing.json',timing)
    geometry = query['geometry']
    summary = dict(name=name,title=title,parent=parent,index=index,image=f'{name}/input.png',
        query_seconds=timing['query_seconds'],geometry=None if geometry is None else geometry['status'],
        inside_surface_pixels=0 if geometry is None else geometry['inside_surface_pixels'],
        checked_pixels=audit['checked_pixels'],remaining_horizon_s=query.get('remaining_horizon_s'),
        evidence_remaining_s=query.get('evidence_remaining_s'),expires_before_horizon=query.get('expires_before_horizon'),
        result_current=final['result_current'],reasons=final['reasons'],flight_authorized=False)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def page(report):
    options,panels=[],[]
    for i,row in enumerate(report['cases']):
        options.append(f'<option value="{i}">{row["title"]}</option>')
        panels.append(f'''<section data-case="{i}" style="display:{'block' if i==0 else 'none'}"><h2>{row['title']}</h2><div class="grid"><figure><img src="{row['image']}" alt="本场景实际使用的输入图像"><figcaption>{'真实 TUM 保存帧；未做导航对齐' if row['name']=='tum-unregistered' else '理想仿真输入；未调用神经网络'}</figcaption></figure><div><p>几何状态：<b>{row['geometry']}</b></p><p>范围内测得表面像素：<b>{row['inside_surface_pixels']}</b>；检查像素：{row['checked_pixels']}</p><p>剩余预测：{row['remaining_horizon_s']} s<br>观测剩余：{row['evidence_remaining_s']} s</p><p>完成时结果仍对应当前上下文：{row['result_current']}；执行许可：否。</p><p>离线查询 {row['query_seconds']*1000:.2f} ms，不包含渲染/观测管线/完成检查。</p></div></div><h3>保留的拒绝原因</h3><ul>{''.join('<li>'+html.escape(r)+'</li>' for r in row['reasons'])}</ul></section>''')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>停止预测与三维深度检查</title><style>body{margin:0;padding:24px;background:#edf2f5;color:#193346;font:16px/1.7 system-ui}main{max-width:1080px;margin:auto}section{background:white;padding:24px;border-radius:12px;margin:20px 0}h1{font-size:30px}select{font:inherit;padding:8px;max-width:100%}.grid{display:grid;grid-template-columns:320px 1fr;gap:28px}figure{margin:0}img{width:100%;image-rendering:pixelated;border-radius:8px}figcaption{font-size:13px}li{overflow-wrap:anywhere}.note{color:#89501e}a{color:#126f88}@media(max-width:720px){.grid{grid-template-columns:1fr}}</style><main><h1>停止轨迹仍匹配，空间依据是否足够？</h1><section><p>12 个固定条件 · 机体范围 + 相对误差 · 原观测期限 · 查询完成后再核对</p><p class="note">这是保存物理轨迹与深度观测的离线组合。未执行新飞行，也没有将空检测、远处采样或期限满足解释为空间安全。TUM 真实输入保持未对齐。</p><label for="case">选择场景：</label><select id="case">'''+''.join(options)+'''</select></section>'''+''.join(panels)+'''<section><h2>上轮时序记录的新核对</h2><p>10 个超期周期均不是接收预测结果的当次周期。3 个为故意卡顿及追赶；其余是 3 次长处理和 4 次随后追赶。没有阶段 CPU/垃圾回收记录，具体根因仍未知。</p><p>几何查询需要独立于快速反馈；本轮没有新的并发计时。</p><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定协议</a> · <a href="dataset-sources.json">TUM 来源与署名</a></p></section></main><script>document.getElementById('case').addEventListener('change',function(){document.querySelectorAll('[data-case]').forEach(e=>e.style.display=e.dataset.case===this.value?'block':'none')})</script></html>'''


def run(output,verify):
    fixed=sources()
    if verify:
        count=check_manifest(output)
        saved=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if saved['sources']!=fixed: raise ValueError('sources changed')
    else:
        output.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(ROOT/'docs/FORECAST_SPACE_PROTOCOL.md',output/'protocol.md')
        shutil.copyfile(DATASET/'sources.json',output/'dataset-sources.json')
    rows=[]
    for spec in CASES:
        folder=output/spec[0]
        if not verify: folder.mkdir()
        rows.append(run_case(spec,folder,verify))
    summary=dict(cases=len(rows),geometry_statuses=dict(Counter(r['geometry'] or 'NOT_QUERIED' for r in rows)),
        checked_pixels=sum(r['checked_pixels'] for r in rows),surface_pixels=sum(r['inside_surface_pixels'] for r in rows),
        current_results=sum(r['result_current'] for r in rows),executed_movements=0,new_model_calls=0)
    report=dict(sources=fixed,parent_manifest_sha256=sha(PARENT/'manifest.json'),
        depth_parent_manifest_sha256=sha(DEPTH_PARENT/'manifest.json'),dataset_manifest_sha256=sha(DATASET/'manifest.json'),
        cases=rows,summary=summary,timing_review=timing_review(),flight_authorized=False,actual_new_physics_steps=0)
    if sources()!=fixed: raise ValueError('sources changed during run')
    if verify:
        if report!=saved or page(report)!=(output/'demo.html').read_text(encoding='utf-8'): raise ValueError('report or page differs')
        if check_manifest(output)!=count: raise ValueError('archive changed')
    else:
        dump(output/'report.json',report)
        (output/'demo.html').write_text(page(report),encoding='utf-8')
        dump(output/'manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
        count=check_manifest(output)
    return dict(verified=verify,files=count,sources=len(fixed),**summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.output.resolve(),args.verify),ensure_ascii=False),flush=True)
