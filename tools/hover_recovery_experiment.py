"""来源：本项目原创。从保存的物理指令重建中止状态，继续控制并确认悬停。"""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from drone_nav.hover_recovery import RecoveryBudget, finish_aborted_operation
from drone_nav.physical_vehicle import PhysicalVehicle, FlightBudget, world_xml
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.realvision import sha, dump
from tools.physical_rescan_experiment import world_for
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT = PROJECT/'work/physical-rescan-02'
SOURCES = ('drone_nav/hover_recovery.py', 'tools/hover_recovery_experiment.py',
           'tests/test_hover_recovery.py', 'drone_nav/hover_recovery_lab.html',
           'docs/HOVER_RECOVERY_PROTOCOL.md')
canonical = lambda v: json.loads(json.dumps(v, allow_nan=False))


def read_parent():
    check_manifest(PARENT)
    report = json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    for name, digest in report['sources'].items():
        if sha(PROJECT/name) != digest:
            raise ValueError('parent source changed: '+name)
    return report


def specs(parent):
    out = [dict(key=c['key'], parent=c['key'], title=c['title'], fault=None, duration=4.)
           for c in parent['cases'] if c['summary']['result']['state'] != 'ARRIVED_AND_SLOW']
    out += [dict(key='short-budget', parent='scan-disturbance', title='恢复时间不足', fault=None, duration=.3),
            dict(key='persistent-thrust', parent='scan-disturbance', title='持续推力异常', fault='thrust', duration=4.),
            dict(key='clock-fault', parent='scan-disturbance', title='恢复反馈时钟冻结', fault='clock', duration=4.)]
    return out


class RecoveryVehicle(PhysicalVehicle):
    recovery_fault = None

    def _step(self, target):
        if self.recovery_fault == 'thrust':
            state = self.apply_motors([8., 8., 8., 8.])
        else:
            state = super()._step(target)
        if self.recovery_fault == 'clock':
            state = dict(state, time_s=state['time_s']-self.physics.dt)
        return state


def run_case(spec, parent_case):
    old = parent_case['summary']
    raw = json.loads(gzip.decompress((PARENT/parent_case['input']).read_bytes()))
    vehicle = RecoveryVehicle(world_for(old['spec']), (1, 5), budget=FlightBudget(**old['budget']))
    try:
        # 逐条重放前段实际电机指令，不用位置重置跳到期望的终点。
        anchor = None
        for i, motors in enumerate(raw['commands'], 1):
            state = vehicle.apply_motors(motors)
            if i == len(raw['commands'])-round(.5/vehicle.physics.dt):
                anchor = tuple(state['position'])
        if vehicle.state() != old['actual_final'] or vehicle.audit != old['audit']:
            raise ValueError('parent physical state/audit mismatch')
        operation = old['result']
        if operation['receipt'] is None:
            anchor = tuple(operation['actual_after_scan']['position'])
        if anchor is None:
            raise ValueError('missing original fixed hover target')
        vehicle.hold_target = anchor
        vehicle.recovery_fault = spec['fault']
        start_count = len(vehicle.commands)
        audit_before = dict(vehicle.audit)
        result = finish_aborted_operation(vehicle, operation,
                                         budget=RecoveryBudget(max_duration_s=spec['duration']))
        after = vehicle.state()
        recovered = result['recovery']
        summary = dict(spec=spec, state=result['state'], original_state=result['original_state'],
                       confirmed_cell=result['confirmed_cell'], recovery={k:v for k,v in recovered.items()
                           if k not in ('states', 'samples')}, actual_final=after,
                       parent_steps=start_count, recovery_steps=len(vehicle.commands)-start_count,
                       parent_audit=audit_before, final_audit=vehicle.audit,
                       additional_clearance_intervals=vehicle.audit['clearance_violations']-audit_before['clearance_violations'],
                       model_sha256=vehicle.physics.model_sha256, dll_sha256=vehicle.physics.dll_sha256,
                       control_actions=len(vehicle.actions), flight_authorized=False, resume_allowed=False)
        evidence = dict(commands=vehicle.commands[start_count:], states=recovered['states'],
                        samples=recovered['samples'])
        return canonical(evidence), canonical(summary)
    finally:
        vehicle.close()


def run(output, verify=False):
    parent = read_parent()
    lookup = {c['key']:c for c in parent['cases']}
    hashes = {n:sha(PROJECT/n) for n in SOURCES}
    if verify:
        count = check_manifest(output)
        report = json.loads((output/'report.json').read_text(encoding='utf-8'))
        if report['sources'] != hashes or report['parent_sha256'] != sha(PARENT/'report.json'):
            raise ValueError('source/parent mismatch')
        if [c['summary']['spec'] for c in report['cases']] != specs(parent):
            raise ValueError('scenario matrix mismatch')
        steps = 0
        for case in report['cases']:
            saved = json.loads(gzip.decompress((output/case['input']).read_bytes()))
            raw, summary = run_case(case['summary']['spec'], lookup[case['summary']['spec']['parent']])
            if raw != saved or summary != case['summary']:
                raise ValueError('feedback replay mismatch: '+case['key'])
            old = lookup[summary['spec']['parent']]
            parent_raw = json.loads(gzip.decompress((PARENT/old['input']).read_bytes()))
            with QuadrotorPhysics(model_xml=world_xml(world_for(old['summary']['spec']))) as physics:
                physics.reset((1.5, 5.5, 3.5))
                for motors in parent_raw['commands']:
                    physics.step(motors)
                actual = [physics.state()]
                for motors in saved['commands']:
                    physics.step(motors); actual.append(physics.state())
                if physics.state() != summary['actual_final']:
                    raise ValueError('independent motor replay mismatch')
                # 协议错帧不进入“已验证状态”序列；实际物理终点仍完整保存核对。
                if actual[:len(saved['states'])] != saved['states']:
                    raise ValueError('verified state prefix mismatch')
            steps += summary['recovery_steps']
        return dict(verified=True, files=count, cases=len(report['cases']), recovery_steps=steps)
    if output.exists():
        raise FileExistsError('refuse overwrite '+str(output))
    output.mkdir(parents=True, exist_ok=False)
    put(output/'protocol.md', (PROJECT/'docs/HOVER_RECOVERY_PROTOCOL.md').read_bytes())
    cases = []
    for spec in specs(parent):
        evidence, summary = run_case(spec, lookup[spec['parent']])
        path = spec['key']+'/recovery.json.gz'
        put(output/path, gzip.compress(json.dumps(evidence, separators=(',', ':')).encode(), mtime=0))
        # 页面保留完整判定曲线，每 0.02 秒抽一帧，仅压缩显示而非压缩判定。
        points = evidence['samples'][::10]
        if points[-1] != evidence['samples'][-1]:
            points.append(evidence['samples'][-1])
        cases.append(dict(key=spec['key'], title=spec['title'], input=path, summary=summary, points=points))
        print(spec['key'], summary['recovery']['state'], summary['recovery_steps'], flush=True)
    if hashes != {n:sha(PROJECT/n) for n in SOURCES}:
        raise ValueError('source changed')
    report = dict(kind='hover-recovery', sources=hashes, parent_sha256=sha(PARENT/'report.json'), cases=cases)
    dump(output/'report.json', report)
    template = (PROJECT/'drone_nav/hover_recovery_lab.html').read_text(encoding='utf-8')
    put(output/'demo.html', template.replace('__DATA__', json.dumps(report, ensure_ascii=False).replace('<', '\\u003c')).encode('utf-8'))
    dump(output/'manifest.json', {p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases), confirmed=sum(c['summary']['recovery']['motion_stable'] for c in cases),
                recovery_steps=sum(c['summary']['recovery_steps'] for c in cases))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.output, a.verify), ensure_ascii=False, indent=2))
