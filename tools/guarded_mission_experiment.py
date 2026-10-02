"""来源：本项目原创。逐段门槛接入完整任务，并保留可用性下降的对照。"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
from html import escape
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from drone_nav.classifier import ColorModel
from drone_nav.descent_experiment import descent_cases
from drone_nav.guarded_mission import GuardedMissionVehicle
from drone_nav.mission_supervisor import MissionSupervisor
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.physical_vehicle import world_xml
from drone_nav.pinhole import Pose
from drone_nav.realvision import sha, dump
from drone_nav.sampling_motion import SamplingAssumption
from drone_nav.verify_coupled import RecordedPhysicalVehicle, saved_world
from drone_nav.verify_descent import replay_raw
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT = PROJECT / 'work/supervised-mission-01'
OWN = ('drone_nav/guarded_mission.py', 'tools/guarded_mission_experiment.py',
       'tests/test_guarded_mission.py', 'docs/GUARDED_MISSION_PROTOCOL.md',
       'drone_nav/physical_rescan.py', 'drone_nav/stationary_rescan.py',
       'drone_nav/sampling_motion.py', 'drone_nav/detection_bridge.py')
canonical = lambda x: json.loads(json.dumps(x, allow_nan=False))


def specs():
    return [dict(key='paved', title='铺装地面', base='paved', size=.2, parent='paved'),
            dict(key='unknown', title='最小尺寸未知', base='paved', size=None),
            dict(key='water', title='水面目的地', base='water', size=.2, parent='water'),
            dict(key='depth', title='巡航深度失效', base='paved', size=.2, dropout=0, parent='cruise-depth'),
            dict(key='ignored', title='控制器忽略指令', base='paved', size=.2, ignored=0, parent='cruise-ignored'),
            dict(key='scan-depth', title='复查深度失效', base='paved', size=.2, fault='depth'),
            dict(key='scan-pose', title='复查位姿错配', base='paved', size=.2, fault='pose')]


class Vehicle(GuardedMissionVehicle):
    def __init__(self, spec, world, saved=None):
        assumption = None if spec['size'] is None else SamplingAssumption(spec['size'], spec['size'])
        super().__init__(world, (3,8), sampling_assumption=assumption,
                         dropout_tick=spec.get('dropout'), ignore_move=spec.get('ignored'))
        self.spec, self.saved = spec, saved
        self.saved_frames = [] if saved is None else saved['frames']
        self.frame_index = self.contact_index = self.rescan_frames = 0

    def capture(self, k, tick, aim=None):
        if self.saved is None:
            frame = super().capture(k, tick, aim)
        else:
            state = self.state()
            frame = RecordedPhysicalVehicle.capture(self, k, tick, aim)
            if self._scan_records is not None:
                self._scan_records.append((state, frame))
        # 原始记录不改写；消费接口故障在初跑和重放的同一位置注入。
        if self.rescanning:
            self.rescan_frames += 1
            if self.rescan_frames == 2:
                if self.spec.get('fault') == 'depth':
                    frame = replace(frame, depth_z_m=(None,)*(k.width*k.height))
                elif self.spec.get('fault') == 'pose':
                    p = frame.pose
                    frame = replace(frame, pose=Pose((p.position[0]+.1,*p.position[1:]),p.right,p.down,p.forward))
        return frame

    def contact_gap(self):
        value = super().contact_gap()
        if self.saved is not None:
            if self.contact_index >= len(self.saved['contact_readings']) or self.contact_readings[-1] != self.saved['contact_readings'][self.contact_index]:
                raise ValueError('probe replay mismatch')
            self.contact_index += 1
        return value


def simulate(spec, model, saved=None):
    world = next(c['world'] for c in descent_cases() if c['key'] == spec['base'])
    v = Vehicle(spec, world, saved)
    try:
        result = MissionSupervisor(v, model, (3,8), (4,8)).run()
        if v.history[-1] != v.state():
            v.history.append(v.state())
        if saved is not None and (v.frame_index != len(saved['frames']) or v.contact_index != len(saved['contact_readings'])):
            raise ValueError('unused saved inputs')
        raw = dict(frames=v.frames if saved is None else saved['frames'], commands=v.commands,
                   contact_readings=v.contact_readings)
        summary = dict(spec=spec, world=asdict(world), result=result, history=v.history,
                       actual_final=v.state(), frames=len(raw['frames']), steps=v.steps,
                       actions=v.actions, audit=v.audit, move_checks=v.move_checks,
                       source_frames=v.source_frames, dll_sha256=v.physics.dll_sha256,
                       model_sha256=v.physics.model_sha256)
        return canonical(raw), canonical(summary)
    finally:
        v.close()


def comparison(cases, parent):
    rows = []
    for case in cases:
        s = case['summary']; key = s['spec'].get('parent')
        old = next((c['summary'] for c in parent['cases'] if c['key']==key), None)
        if old is not None and old['world'] != s['world']:
            raise ValueError('comparison world changed')
        rows.append(dict(title=case['title'], old_state=None if old is None else old['result']['state'],
                         old_moves=None if old is None else len(old['actions']),
                         new_state=s['result']['state'], new_moves=len(s['actions']),
                         reason=s['result']['original_reason']))
    return rows


def page(report):
    rows = ''.join('<tr>'+''.join('<td>'+escape(str(v) if v is not None else '无同阶段对照')+'</td>' for v in
                   (r['title'],r['old_state'],r['old_moves'],r['new_state'],r['new_moves'],r['reason']))+'</tr>'
                   for r in report['comparison'])
    details = []
    for case in report['cases']:
        s=case['summary']; body=[]
        for q in s['move_checks']:
            g=q['final_guard']; reasons=Counter(reason for row in g['sampling'] for c in row['candidates'] for reason in c['reasons'])
            body.append(dict(state=q['state'], baseline=g['baseline'], required_cells=g['required_cells'],
                             supported_cells=[r['voxel'] for r in g['sampling'] if r['supported']],
                             candidate_rejection_counts=dict(reasons), rescan_frames=len(q['events'])))
        details.append('<details><summary>'+escape(case['title'])+'：逐格检查理由</summary><pre>'+escape(json.dumps(body,ensure_ascii=False,indent=2))+'</pre></details>')
    return ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>采样门槛接入主导航</title>
<style>body{margin:0;background:#eef3f4;color:#243f48;font:16px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1160px;margin:auto;padding:28px 22px}h1{font-size:28px}p{line-height:1.8}.notice{background:#fff0df;border-left:4px solid #aa6622;padding:16px}.table{overflow:auto;background:white}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:12px;border-bottom:1px solid #dce4e8;text-align:left}details{background:white;margin:12px 0;padding:18px;border-radius:8px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}a{color:#176c79}</style>
<main><h1>采样门槛已接入，首段仍缺观测支持</h1><p>沿用统一任务主管和物理控制器；每次移动前执行四帧悬停复查。以下是实际任务结果，保留新条件带来的任务失败。</p>
<p class="notice">七例都未发出航点移动。原铺装场景曾确认接地，新入口在起点中止；这表明当前整格覆盖条件限制了可用性，不能称为避障性能提高。真实图像识别仍未接入控制。</p>
<div class="table"><table><thead><tr><th>场景</th><th>原终态</th><th>原移动数</th><th>新终态</th><th>新移动数</th><th>保留原因</th></tr></thead><tbody>'''+rows+'''</tbody></table></div>
<p>“运动稳定”只表示中止后的状态持续达标，不表示到达、空间安全或外卖交付。下方计数是“帧—格候选”的拒绝理由数量，不是障碍个数；基线 APPROVED 也不证明整个格子已被相机看全。</p>'''+''.join(details)+'''<p><a href="report.json">完整报告</a> · <a href="protocol.md">实验协议</a> · <a href="manifest.json">文件摘要</a></p></main></html>''').encode('utf-8')


def run(output, verify=False):
    check_manifest(PARENT)
    parent=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    hashes={n:sha(PROJECT/n) for n in set(parent['sources'])|set(OWN)}
    for n,d in parent['sources'].items():
        if hashes[n]!=d:raise ValueError('parent source changed: '+n)
    model=ColorModel.from_dict(json.loads((PARENT/'model.json').read_text(encoding='utf-8')))
    if verify:
        count=check_manifest(output); report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if report['sources']!=hashes or report['parent_sha256']!=sha(PARENT/'report.json'):
            raise ValueError('source/parent mismatch')
        if [c['summary']['spec'] for c in report['cases']]!=specs():raise ValueError('case matrix mismatch')
        for case in report['cases']:
            raw=json.loads(gzip.decompress((output/case['input']).read_bytes()))
            new,summary=simulate(case['summary']['spec'],model,raw)
            if new!=raw or summary!=case['summary']:raise ValueError('task replay mismatch')
            with QuadrotorPhysics(model_xml=world_xml(saved_world(summary['world']))) as physics:
                if replay_raw(physics,raw,(3,8))!=summary['history'] or physics.state()!=summary['actual_final']:
                    raise ValueError('motor replay mismatch')
        if comparison(report['cases'],parent)!=report['comparison']:raise ValueError('comparison mismatch')
        if page(report)!=(output/'demo.html').read_bytes():raise ValueError('page mismatch')
        return dict(verified=True,files=count,cases=len(report['cases']),
                    frames=sum(c['summary']['frames'] for c in report['cases']),
                    steps=sum(c['summary']['steps'] for c in report['cases']))
    if output.exists():raise FileExistsError('refuse overwrite '+str(output))
    output.mkdir(parents=True)
    cases=[]
    for spec in specs():
        raw,summary=simulate(spec,model)
        name=spec['key']+'/raw.json.gz'
        put(output/name,gzip.compress(json.dumps(raw,separators=(',',':')).encode(),mtime=0))
        cases.append(dict(key=spec['key'],title=spec['title'],input=name,summary=summary))
        print(spec['key'],summary['result']['original_reason'],len(summary['actions']),flush=True)
    if hashes!={n:sha(PROJECT/n) for n in hashes}:raise ValueError('source changed during run')
    report=dict(kind='guarded-mission',sources=hashes,parent_sha256=sha(PARENT/'report.json'),
                cases=cases,comparison=comparison(cases,parent))
    dump(output/'report.json',report)
    put(output/'protocol.md',(PROJECT/'docs/GUARDED_MISSION_PROTOCOL.md').read_bytes())
    put(output/'demo.html',page(report))
    dump(output/'manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases),frames=sum(c['summary']['frames'] for c in cases),steps=sum(c['summary']['steps'] for c in cases))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.verify),ensure_ascii=False,indent=2))
