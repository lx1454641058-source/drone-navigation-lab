"""来源：本项目原创。显式测距状态接入开放场地任务的开发对照。"""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
from html import escape
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from drone_nav.classifier import ColorModel
from drone_nav.mission_supervisor import MissionSupervisor
from drone_nav.range_observation import RangeFrame, RangeSamplingHistory, render_range, legacy_frame
from drone_nav.pinhole import Pose
from drone_nav.physical_vehicle import rotate_vector, world_xml
from drone_nav.verify_exploration import RecordedCamera
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.verify_coupled import saved_world
from drone_nav.verify_descent import replay_raw
from drone_nav.realvision import sha, dump
from tools.active_observation_experiment import Vehicle as ActiveVehicle, specs as prior_specs
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT=PROJECT/'work/active-observation-01'
MODEL=PROJECT/'work/supervised-mission-01/model.json'
OWN=('drone_nav/range_observation.py','tools/range_observation_experiment.py',
     'tests/test_range_observation.py','docs/RANGE_OBSERVATION_PROTOCOL.md')
canonical=lambda x:json.loads(json.dumps(x,allow_nan=False))


def specs():
    base=next(s for s in prior_specs() if s['key']=='active')
    cases=[('legacy','开放场地：旧空值语义',dict(explicit=False)),
           ('open','开放场地：显式测距状态',{}),
           ('closed','封闭场地回归',dict(enclosed=True)),
           ('invalid','相机全部测距失效',dict(range_fault='all')),
           ('probe-invalid','补拍期间测距失效',dict(range_fault='probe')),
           ('short-range','量程缩短到 3 米',dict(range_m=3.)),
           ('obstacle','可见 20 厘米障碍',dict(obstacle=.2)),
           ('thin','2 毫米障碍反例',dict(obstacle=.002)),
           ('unknown-size','最小障碍尺寸未知',dict(size=None))]
    return [dict(base,key=key,title=title,**dict(dict(enclosed=False,explicit=True,
        range_m=30.,range_fault=None),**changes)) for key,title,changes in cases]


class Vehicle(ActiveVehicle):
    def __init__(self,s,saved=None):
        super().__init__(s,saved)
        h=self.sampling_history
        if s['explicit']:
            self.sampling_history=RangeSamplingHistory(divisions=h.divisions,initial_region=h.initial_region)

    def capture(self,k,tick,aim=None):
        state=self.state();xyz=tuple(state['position']);q=state['quaternion']
        base=Pose.look_at(xyz,(xyz[0],xyz[1],0)) if aim is None else aim(xyz)
        pose=Pose(xyz,rotate_vector(base.right,q),rotate_vector(base.down,q),rotate_vector(base.forward,q))
        if self.saved is None:
            probing=bool(self.probes and self.probes[-1]['status']=='OBSERVING')
            invalid=self.spec['range_fault']=='all' or (self.spec['range_fault']=='probe' and probing)
            frame=render_range(self.world,pose,k,tick=tick,max_range_m=self.spec['range_m'],invalid=invalid)
            self.frames.append(dict(frame=asdict(frame),capture_state=state))
        else:
            if self.frame_index>=len(self.saved_frames):raise ValueError('recorded range frames exhausted')
            item=self.saved_frames[self.frame_index]
            if state!=item['capture_state']:raise ValueError('range capture state mismatch')
            old=RecordedCamera([item['frame']]).capture(pose,k,tick)
            data=item['frame']
            frame=RangeFrame(old.intrinsics,old.pose,old.rgb,old.depth_z_m,old.tick,
                tuple(data['outcomes']),data['max_range_m'],data['sensor_model'])
            frame.validate();self.frame_index+=1
        # 原始传感器输出不变；旧方案只消费兼容的深度字段。
        consumed=frame if self.spec['explicit'] else legacy_frame(frame)
        if self._scan_records is not None:self._scan_records.append((state,consumed))
        return consumed


def simulate(s,model,saved=None):
    v=Vehicle(s,saved)
    try:
        result=MissionSupervisor(v,model,(3,8),(5,8),max_ticks=4).run()
        if v.history[-1]!=v.state():v.history.append(v.state())
        frames=v.frames if saved is None else saved['frames']
        if saved is not None and (v.frame_index!=len(frames) or v.contact_index!=len(saved['contact_readings'])):
            raise ValueError('unused range observations')
        raw=dict(frames=frames,commands=v.commands,contact_readings=v.contact_readings)
        outcomes=Counter(x for item in frames for x in item['frame']['outcomes'])
        summary=dict(spec=s,result=result,world=asdict(v.world),history=v.history,actual_final=v.state(),
            frames=len(frames),steps=v.steps,actions=v.actions,audit=v.audit,move_checks=v.move_checks,
            source_frames=v.source_frames,probes=v.probes,range_outcomes=dict(outcomes),
            initial_region=asdict(v.sampling_history.initial_region),dll_sha256=v.physics.dll_sha256,
            model_sha256=v.physics.model_sha256)
        return canonical(raw),canonical(summary)
    finally:v.close()


def page(report):
    rows=[];details=[]
    for c in report['cases']:
        s=c['summary'];r=s['result']
        values=(c['title'],r['state'],r['confirmed_cell'],s['frames'],s['audit']['clearance_violations'],r['original_reason'] or '接地确认')
        rows.append('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in values)+'</tr>')
        guards=[]
        for check in s['move_checks']:
            g=check['final_guard']
            if g is None:continue
            patches=[p for row in g['sampling'] for p in row['patches']]
            guards.append(dict(baseline=g['baseline'],boxes=len(patches),supported=sum(p['source'] is not None for p in patches),
                range_supported=sum(bool(p['source'] and p['source'].get('range_contract')) for p in patches)))
        details.append('<details><summary>'+escape(c['title'])+'：来源与检查</summary><pre>'+escape(json.dumps(
            dict(spec=s['spec'],outcomes=s['range_outcomes'],guards=guards),ensure_ascii=False,indent=2))+'</pre></details>')
    return ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>开放场地的测距状态</title>
<style>body{margin:0;background:#edf3f5;color:#183944;font:16px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:28px}p{line-height:1.8}.notice{background:#fff0d9;padding:16px;border-left:4px solid #ac7332}.table{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:13px}td,th{padding:12px;border-bottom:1px solid #dbe5e9;text-align:left}details{background:white;margin:12px 0;padding:16px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#176277}</style>
<main><h1>开放场地：没有命中，还是测距失效？</h1><p>HIT 保留真实命中深度；NO_HIT 只提供有限射线长度内未命中的模拟结果；INVALID 保持未知。普通空深度不能推断为 NO_HIT，也不创建虚假表面点。</p>
<p class="notice">理想中心射线、精确位姿、静态场地、明确起始区域和最小障碍尺寸假设仍然存在。开放场景的条件性成功不代表真实视觉避障；请一起查看细障碍反例。页面展示保存结果，不是实时飞行。</p>
<div class="table"><table><thead><tr><th>条件</th><th>任务终态</th><th>确认格</th><th>图像帧</th><th>空间侵犯区间</th><th>原因</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>'''+''.join(details)+'''
<p>未命中不写入旧地图；旧地图和物理门槛继续检查实际命中证据。接地不等于配送完成。</p><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定协议</a> · <a href="manifest.json">文件摘要</a></p></main></html>''').encode('utf-8')


def run(output,verify=False):
    check_manifest(PARENT);parent=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    hashes={n:sha(PROJECT/n) for n in sorted(set(parent['sources'])|set(OWN))}
    for n,d in parent['sources'].items():
        if hashes[n]!=d:raise ValueError('parent source changed')
    check_manifest(MODEL.parent)
    model=ColorModel.from_dict(json.loads(MODEL.read_text(encoding='utf-8')))
    if verify:
        files=check_manifest(output);report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if report['sources']!=hashes or report['parent_sha256']!=sha(PARENT/'report.json') or report['model_sha256']!=sha(MODEL):raise ValueError('source/input mismatch')
        if [c['summary']['spec'] for c in report['cases']]!=specs():raise ValueError('matrix mismatch')
        for c in report['cases']:
            raw=json.loads(gzip.decompress((output/c['input']).read_bytes()))
            new,s=simulate(c['summary']['spec'],model,raw)
            if new!=raw or s!=c['summary']:raise ValueError('range replay mismatch '+c['key'])
            # 独立从保存场景/真实采集位姿重新生成每个传感器输出，避免只重放
            # 自报的 NO_HIT 标签。故障由协议与实际主动阶段时间确定。
            world=saved_world(s['world']);spec=s['spec']
            from drone_nav.pinhole import Intrinsics
            for item in raw['frames']:
                data=item['frame'];t=item['capture_state']['time_s']
                probing=any(p['actual_start']['time_s']<=t<p['actual_final']['time_s'] for p in s['probes'])
                invalid=spec['range_fault']=='all' or (spec['range_fault']=='probe' and probing)
                pose=Pose(**{name:tuple(value) for name,value in data['pose'].items()})
                frame=render_range(world,pose,Intrinsics(**data['intrinsics']),tick=data['tick'],
                    max_range_m=spec['range_m'],invalid=invalid)
                if canonical(asdict(frame))!=data:raise ValueError('sensor rerender mismatch '+c['key'])
            with QuadrotorPhysics(model_xml=world_xml(saved_world(s['world']))) as physics:
                if replay_raw(physics,raw,(3,8))!=s['history'] or physics.state()!=s['actual_final']:raise ValueError('motor replay mismatch')
            print('verified',c['key'],flush=True)
        if page(report)!=(output/'demo.html').read_bytes():raise ValueError('page mismatch')
        return dict(verified=True,files=files,cases=len(report['cases']))
    if output.exists():raise FileExistsError('refuse overwrite '+str(output))
    output.mkdir(parents=True);cases=[]
    for s in specs():
        raw,summary=simulate(s,model);name=s['key']+'/raw.json.gz'
        put(output/name,gzip.compress(json.dumps(raw,separators=(',',':')).encode(),mtime=0))
        cases.append(dict(key=s['key'],title=s['title'],input=name,summary=summary))
        print(s['key'],summary['result']['state'],summary['result']['original_reason'],flush=True)
    if hashes!={n:sha(PROJECT/n) for n in hashes}:raise ValueError('source changed during run')
    report=dict(kind='range-observation',sources=hashes,parent_sha256=sha(PARENT/'report.json'),model_sha256=sha(MODEL),cases=cases)
    dump(output/'report.json',report);put(output/'demo.html',page(report));put(output/'protocol.md',(PROJECT/'docs/RANGE_OBSERVATION_PROTOCOL.md').read_bytes())
    dump(output/'manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases),frames=sum(c['summary']['frames'] for c in cases),steps=sum(c['summary']['steps'] for c in cases))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
