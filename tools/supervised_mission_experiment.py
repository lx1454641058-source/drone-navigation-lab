"""来源：本项目原创。统一任务端到端开发实验与双路径物理重放。"""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from drone_nav.classifier import ColorModel
from drone_nav.descent_experiment import descent_cases
from drone_nav.mission_supervisor import MissionSupervisor, MissionVehicle
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.physical_vehicle import world_xml
from drone_nav.verify_coupled import RecordedPhysicalVehicle
from drone_nav.verify_descent import replay_raw
from drone_nav.visual_descent import DescentConfig
from drone_nav.raycast import Surface, World
from drone_nav.realvision import sha, dump
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT = PROJECT/'work/hover-recovery-01'
MODEL = PROJECT/'work/run-v11-descent-validated/model.json'
SOURCES = ('drone_nav/mission_supervisor.py', 'tools/supervised_mission_experiment.py',
           'tests/test_mission_supervisor.py', 'drone_nav/supervised_mission_lab.html',
           'docs/SUPERVISED_MISSION_PROTOCOL.md', 'drone_nav/physical_navigation.py',
           'drone_nav/visual_descent.py', 'drone_nav/descent_vehicle.py',
           'drone_nav/hover_recovery.py', 'drone_nav/physical_vehicle.py',
           'drone_nav/native_physics.py', 'drone_nav/quadrotor.xml', 'drone_nav/flight_control.py',
           'drone_nav/perspective_landing.py', 'drone_nav/classifier.py', 'drone_nav/raycast.py',
           'drone_nav/pinhole.py', 'drone_nav/occupancy.py', 'drone_nav/reachable_policy.py',
           'drone_nav/verify_coupled.py', 'drone_nav/verify_descent.py')
canonical = lambda x:json.loads(json.dumps(x, allow_nan=False))


def specs():
    return [dict(key='paved', title='正常到达与接地', base='paved'),
            dict(key='water', title='水面拒降后恢复', base='water'),
            dict(key='cruise-depth', title='巡航首批深度失效', base='paved', dropout=0),
            dict(key='cruise-ignored', title='巡航指令未执行', base='paved', ignored=0),
            dict(key='bump', title='同色凸起导致下降拒绝', base='bump'),
            dict(key='descent-depth', title='下降途中深度失效', base='depth'),
            dict(key='short-abort', title='下降中止后继续恢复', base='paved', short_abort=True),
            dict(key='post-disarm', title='停桨后接地探针失效', base='postcontact'),
            dict(key='recovery-fault', title='恢复期间持续推力异常', base='water', recovery_fault=True)]


class ExperimentVehicle(MissionVehicle):
    def __init__(self, spec, world, saved=None, **kwargs):
        super().__init__(world, (3, 8), dropout_tick=spec.get('dropout'),
                         ignore_move=spec.get('ignored'), **kwargs)
        self.spec, self.saved = spec, saved
        self.saved_frames = [] if saved is None else saved['frames']
        self.frame_index = self.contact_index = 0

    def capture(self, k, tick, aim=None):
        if self.saved is not None:
            return RecordedPhysicalVehicle.capture(self, k, tick, aim)
        return super().capture(k, tick, aim)

    def contact_gap(self):
        value = super().contact_gap()
        if self.saved is not None:
            if self.contact_index >= len(self.saved['contact_readings']):
                raise ValueError('replay probe exhausted')
            if self.contact_readings[-1] != self.saved['contact_readings'][self.contact_index]:
                raise ValueError('replay probe differs')
            self.contact_index += 1
        return value

    def apply_motors(self, motors):
        if self.spec.get('recovery_fault') and self.operation_phase=='RECOVERING':
            motors = [8., 8., 8., 8.]
        return super().apply_motors(motors)


def simulate_case(spec, model, saved=None):
    case = next(c for c in descent_cases() if c['key']==spec['base'])
    config = DescentConfig(max_duration_s=.5, abort_hold_s=.1) if spec.get('short_abort') else case.get('config', DescentConfig())
    vehicle = ExperimentVehicle(spec, case['world'], saved=saved, fault=case.get('fault'),
                                fault_after_s=case.get('fault_after_s', 4.))
    try:
        supervisor = MissionSupervisor(vehicle, model, (3, 8), (4, 8), descent_config=config)
        result = supervisor.run()
        if vehicle.history[-1] != vehicle.state():
            vehicle.history.append(vehicle.state())
        raw = dict(frames=vehicle.frames if saved is None else saved['frames'], commands=vehicle.commands,
                   contact_readings=vehicle.contact_readings)
        if saved is not None and (vehicle.frame_index != len(saved['frames']) or
                                 vehicle.contact_index != len(saved['contact_readings'])):
            raise ValueError('unconsumed replay inputs')
        summary = dict(spec=spec, world=asdict(case['world']), result=result, history=vehicle.history,
                       actions=vehicle.actions, steps=vehicle.steps, frames=len(raw['frames']),
                       audit=vehicle.audit, actual_final=vehicle.state(),
                       dll_sha256=vehicle.physics.dll_sha256, model_sha256=vehicle.physics.model_sha256)
        return canonical(raw), canonical(summary)
    finally:
        vehicle.close()


def run(output, verify=False):
    check_manifest(PARENT)
    parent = json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    for name, digest in parent['sources'].items():
        if sha(PROJECT/name) != digest:
            raise ValueError('parent source changed')
    old_manifest = json.loads((MODEL.parent/'manifest.json').read_text(encoding='utf-8'))
    expected = next(e['sha256'] for e in old_manifest if e['path']=='model.json')
    if sha(MODEL) != expected:
        raise ValueError('original color model archive digest mismatch')
    model_data = json.loads(MODEL.read_text(encoding='utf-8'))
    model = ColorModel.from_dict(model_data)
    hashes = {n:sha(PROJECT/n) for n in SOURCES}
    if verify:
        count = check_manifest(output)
        report = json.loads((output/'report.json').read_text(encoding='utf-8'))
        if (report['sources'] != hashes or report['parent_sha256'] != sha(PARENT/'report.json')
                or report['color_model_sha256'] != sha(MODEL)):
            raise ValueError('source or input model changed')
        if [c['summary']['spec'] for c in report['cases']] != specs():
            raise ValueError('incomplete experiment matrix')
        steps = frames = 0
        from drone_nav.verify_coupled import saved_world
        for case in report['cases']:
            raw = json.loads(gzip.decompress((output/case['input']).read_bytes()))
            replay, summary = simulate_case(case['summary']['spec'], model, raw)
            if replay != raw or summary != case['summary']:
                raise ValueError('decision/physics replay mismatch: '+case['key'])
            with QuadrotorPhysics(model_xml=world_xml(saved_world(summary['world']))) as physics:
                history = replay_raw(physics, raw, (3, 8))
                if history != summary['history'] or physics.state() != summary['actual_final']:
                    raise ValueError('independent motor/probe replay mismatch')
            steps += summary['steps']; frames += summary['frames']
        return dict(verified=True, files=count, cases=len(report['cases']), frames=frames, physics_steps=steps)
    if output.exists():
        raise FileExistsError('refuse overwrite '+str(output))
    output.mkdir(parents=True, exist_ok=False)
    put(output/'protocol.md', (PROJECT/'docs/SUPERVISED_MISSION_PROTOCOL.md').read_bytes())
    put(output/'model.json', MODEL.read_bytes())
    cases = []
    for spec in specs():
        raw, summary = simulate_case(spec, model)
        name = spec['key']+'/raw.json.gz'
        put(output/name, gzip.compress(json.dumps(raw, separators=(',', ':')).encode(), mtime=0))
        cases.append(dict(key=spec['key'], title=spec['title'], input=name, summary=summary))
        print(spec['key'], summary['result']['state'], summary['steps'], flush=True)
    if hashes != {n:sha(PROJECT/n) for n in SOURCES}:
        raise ValueError('source changed during experiment')
    report = dict(kind='supervised-mission', sources=hashes, parent_sha256=sha(PARENT/'report.json'),
                  color_model_sha256=sha(MODEL), cases=cases)
    dump(output/'report.json', report)
    template = (PROJECT/'drone_nav/supervised_mission_lab.html').read_text(encoding='utf-8')
    put(output/'demo.html', template.replace('__DATA__', json.dumps(report, ensure_ascii=False).replace('<', '\\u003c')).encode('utf-8'))
    dump(output/'manifest.json', {p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases), frames=sum(c['summary']['frames'] for c in cases),
                physics_steps=sum(c['summary']['steps'] for c in cases))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    print(json.dumps(run(args.output, args.verify), ensure_ascii=False, indent=2))
