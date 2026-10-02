"""来源：本项目原创。物理飞行获得历史帧，再复查、执行或拒绝下一段。"""
import argparse
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.occupancy import VoxelMap
from drone_nav.physical_rescan import PhysicalSamplingHistory, rescan_and_move
from drone_nav.physical_vehicle import PhysicalVehicle, FlightBudget, physical_guard, world_xml
from drone_nav.pinhole import Intrinsics, Pose
from drone_nav.sampling_motion import SamplingView, SamplingAssumption
from drone_nav.raycast import World, Surface, Box, render
from drone_nav.realvision import sha, dump
from drone_nav.imaging import png_bytes
from drone_nav.verify_coupled import RecordedPhysicalVehicle
from tools.object_probe import check_manifest
from tools.setup_vision import put

SOURCES = ('drone_nav/physical_rescan.py', 'tools/physical_rescan_experiment.py',
           'tests/test_physical_rescan.py', 'drone_nav/physical_rescan_lab.html',
           'drone_nav/stationary_rescan.py', 'drone_nav/sampling_motion.py',
           'drone_nav/physical_vehicle.py', 'drone_nav/native_physics.py',
           'drone_nav/quadrotor.xml', 'drone_nav/flight_control.py',
           'drone_nav/occupancy.py', 'drone_nav/pinhole.py', 'drone_nav/raycast.py',
           'drone_nav/localization.py', 'drone_nav/verify_coupled.py',
           'drone_nav/verify_exploration.py', 'drone_nav/exploration.py',
           'drone_nav/rendering.py', 'drone_nav/imaging.py',
           'docs/PHYSICAL_RESCAN_PROTOCOL.md')
canonical = lambda x: json.loads(json.dumps(x, allow_nan=False))


def specs():
    result = []
    def add(key, title, **changes):
        item = dict(key=key, title=title, ttl=20., thin=0., offset=0., min_size=.2,
                    history=True, mode='phase_diverse', settle=.1, fault=None)
        item.update(changes)
        result.append(item)
    add('default-age', '原默认 12 秒有效期', ttl=12.)
    add('static-20s', '静态场景 20 秒条件对照')
    add('thin-repeat', '2 厘米障碍，同向重复', thin=.02, mode='repeat')
    add('thin-diverse', '2 厘米障碍，改变角度', thin=.02)
    add('tiny-diverse', '偏移 2 毫米障碍', thin=.002, offset=.007)
    add('unknown-size', '尺寸假设未知', min_size=None)
    add('no-history', '不提供早期历史', history=False)
    add('depth-loss', '复查第二帧深度丢失', fault='depth')
    add('pose-mismatch', '复查第二帧位姿错误', fault='pose')
    add('slow-scan', '慢扫描消耗有效期', settle=2.95)
    add('ignored-command', '控制器不执行第二段', fault='ignored')
    add('scan-disturbance', '复查期间短时推力异常', fault='thrust')
    return result


def world_for(spec):
    items = [Surface('ground', 0, (-40, 60, -40, 56)),
             Box('wall', 3, (12, -40, 0), (12.2, 60, 30))]
    if spec['thin']:
        half = spec['thin']/2
        center = (5.2, 5.5+spec['offset'], 3.5+spec['offset'])
        items.append(Box('thin', 4, tuple(x-half for x in center), tuple(x+half for x in center)))
    return World(tuple(items))


class ExperimentVehicle(PhysicalVehicle):
    def __init__(self, spec, saved=None):
        super().__init__(world_for(spec), (1, 5), budget=FlightBudget(free_ttl_s=spec['ttl']),
                         ignore_move=1 if spec['fault']=='ignored' else None)
        self.spec, self.saved_frames = spec, saved
        self.frame_index = 0
        self.rescanning = False
        self.rescan_frames = 0
        self.thrust_fault_steps = 0

    def capture(self, k, tick, aim=None):
        if self.saved_frames is not None:
            frame = RecordedPhysicalVehicle.capture(self, k, tick, aim)
        else:
            frame = super().capture(k, tick, aim)
        # 保存的是传感器原始帧；下列故障属于消费接口，重放时按相同条件再次注入。
        if self.rescanning:
            index = self.rescan_frames
            self.rescan_frames += 1
            if index == 1 and self.spec['fault'] == 'depth':
                frame = replace(frame, depth_z_m=(None,)*(k.width*k.height))
            if index == 1 and self.spec['fault'] == 'pose':
                pose = frame.pose
                frame = replace(frame, pose=Pose((pose.position[0]+.1, *pose.position[1:]),
                                                pose.right, pose.down, pose.forward))
        return frame

    def apply_motors(self, motors):
        if self.rescanning and self.spec['fault']=='thrust' and self.thrust_fault_steps < 8:
            motors = [8., 8., 8., 8.]
            self.thrust_fault_steps += 1
        return super().apply_motors(motors)


def simulate_case(spec, saved=None):
    vehicle = ExperimentVehicle(spec, None if saved is None else saved['frames'])
    try:
        grid = VoxelMap(20, 16)
        stamps = {0: vehicle.state()['time_s']}
        frames = vehicle.scan(0, (4, 5))
        grid.integrate_moving(frames, 0)
        k = Intrinsics(160, 120, 50, 50, 79.5, 59.5)
        capture = vehicle.state()['time_s']
        frame = vehicle.capture(k, 1, lambda xyz: Pose.look_at(xyz, (xyz[0]+1, xyz[1], xyz[2])))
        vehicle.hold(.05)
        available = vehicle.state()['time_s']
        stamps[1] = capture
        grid.integrate_moving([frame], 1)
        history = PhysicalSamplingHistory(clock_id='physics-seconds')
        if spec['history']:
            history.add(SamplingView(frame, 'approach-history', 'camera', 'physics-seconds',
                                     capture, available), now_s=available)
        warmup_guard = physical_guard(grid, (1, 5), (4, 5), 1, available, stamps, vehicle.budget)
        approach = vehicle.move((4, 5)) if warmup_guard['allowed'] else None
        result = None
        if approach and approach['completed']:
            vehicle.rescanning = True
            a = None if spec['min_size'] is None else SamplingAssumption(spec['min_size'], spec['min_size'])
            result = rescan_and_move(vehicle, history, (4, 5), (5, 5), Intrinsics(40, 30, 25, 25, 19.5, 14.5),
                                     assumption=a, mode=spec['mode'], settle_s=spec['settle'])
        if vehicle.history[-1] != vehicle.state():
            vehicle.history.append(vehicle.state())
        if saved is not None and vehicle.frame_index != len(saved['frames']):
            raise ValueError('unused camera records')
        recorded = vehicle.frames if saved is None else saved['frames']
        summary = dict(spec=spec, warmup_guard=warmup_guard, approach=approach, result=result,
                       engine_version=vehicle.physics.version, dll_sha256=vehicle.physics.dll_sha256,
                       model_sha256=vehicle.physics.model_sha256, history=vehicle.history,
                       steps=vehicle.steps, frame_count=9+vehicle.rescan_frames, audit=vehicle.audit,
                       actual_final=vehicle.state(), actions=vehicle.actions,
                       historical_capture_s=capture, budget=asdict(vehicle.budget),
                       camera_metadata=[item['capture_state'] for item in recorded])
        raw = dict(frames=recorded, commands=vehicle.commands)
        return canonical(raw), canonical(summary)
    finally:
        vehicle.close()


def run(output, verify=False):
    parent = PROJECT/'work/stationary-rescan-01'
    check_manifest(parent)
    parent_report = json.loads((parent/'report.json').read_text(encoding='utf-8'))
    for name, digest in parent_report['sources'].items():
        if sha(PROJECT/name) != digest:
            raise ValueError('parent source changed: '+name)
    hashes = {name: sha(PROJECT/name) for name in SOURCES}
    if verify:
        count = check_manifest(output)
        report = json.loads((output/'report.json').read_text(encoding='utf-8'))
        if (hashes != report['sources'] or sha(output/'protocol.md') != report['protocol_sha256']
                or report['parent_sha256'] != sha(parent/'report.json')):
            raise ValueError('source or protocol mismatch')
        total_steps = total_frames = 0
        for case in report['cases']:
            raw = json.loads(gzip.decompress((output/case['input']).read_bytes()))
            replay, summary = simulate_case(case['summary']['spec'], raw)
            if replay != raw or summary != case['summary']:
                raise ValueError('camera/control replay mismatch: '+case['key'])
            spec = summary['spec']
            for item, image in zip(raw['frames'], case['images'], strict=True):
                data = item['frame']
                pose = Pose(**{k: tuple(v) for k, v in data['pose'].items()})
                k = Intrinsics(**data['intrinsics'])
                frame, _ = render(world_for(spec), pose, k, tick=data['tick'], seed=1701)
                if canonical(asdict(frame)) != data:
                    raise ValueError('raw camera rerender mismatch')
                if png_bytes(k.width, k.height, frame.rgb) != (output/image).read_bytes():
                    raise ValueError('camera PNG mismatch')
            with QuadrotorPhysics(model_xml=world_xml(world_for(spec))) as physics:
                physics.reset((1.5, 5.5, 3.5))
                states = [physics.state()]
                for i, motors in enumerate(raw['commands'], 1):
                    physics.step(motors)
                    if i % 10 == 0:
                        states.append(physics.state())
                if states[-1] != physics.state():
                    states.append(physics.state())
                if states != summary['history'] or physics.state() != summary['actual_final']:
                    raise ValueError('independent thrust replay mismatch')
            total_steps += summary['steps']
            total_frames += summary['frame_count']
        return dict(verified=True, files=count, cases=len(report['cases']),
                    replayed_frames=total_frames, motor_steps=total_steps)
    if output.exists():
        raise FileExistsError('refuse overwrite '+str(output))
    output.mkdir(parents=True, exist_ok=False)
    put(output/'protocol.md', (PROJECT/'docs/PHYSICAL_RESCAN_PROTOCOL.md').read_bytes())
    cases = []
    for spec in specs():
        raw, summary = simulate_case(spec)
        name = spec['key']+'/input.json.gz'
        put(output/name, gzip.compress(json.dumps(raw, separators=(',', ':')).encode(), mtime=0))
        images = []
        for i, item in enumerate(raw['frames']):
            f = item['frame']; k = f['intrinsics']
            img = spec['key']+f'/frame-{i}.png'
            put(output/img, png_bytes(k['width'], k['height'], f['rgb']))
            images.append(img)
        cases.append(dict(key=spec['key'], title=spec['title'], input=name, images=images, summary=summary))
        print(spec['key'], summary['result']['state'] if summary['result'] else 'APPROACH_HOLD',
              summary['steps'], flush=True)
    if hashes != {name: sha(PROJECT/name) for name in SOURCES}:
        raise ValueError('source changed during experiment')
    report = dict(kind='physical-rescan', sources=hashes, protocol_sha256=sha(output/'protocol.md'),
                  parent_sha256=sha(parent/'report.json'), cases=cases)
    dump(output/'report.json', report)
    template = (PROJECT/'drone_nav/physical_rescan_lab.html').read_text(encoding='utf-8')
    put(output/'demo.html', template.replace('__DATA__', json.dumps(report, ensure_ascii=False).replace('<', '\\u003c')).encode('utf-8'))
    dump(output/'manifest.json', {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases), frames=sum(c['summary']['frame_count'] for c in cases),
                motor_steps=sum(c['summary']['steps'] for c in cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.verify), ensure_ascii=False, indent=2))
