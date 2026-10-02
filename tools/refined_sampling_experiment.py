"""来源：本项目原创。近场细分任务实验和保存输入/独立物理重放。"""
import argparse
from dataclasses import asdict
import gzip
from html import escape
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from drone_nav.classifier import ColorModel
from drone_nav.mission_supervisor import MissionSupervisor
from drone_nav.refined_sampling import RefinedMissionVehicle, InitialClearRegion
from drone_nav.sampling_motion import SamplingAssumption
from drone_nav.raycast import World,Surface,Box
from drone_nav.physical_vehicle import world_xml
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.verify_coupled import RecordedPhysicalVehicle,saved_world
from drone_nav.verify_descent import replay_raw
from drone_nav.realvision import sha,dump
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT=PROJECT/'work/guarded-mission-01'
MODEL=PROJECT/'work/supervised-mission-01/model.json'
OWN=('drone_nav/refined_sampling.py','tools/refined_sampling_experiment.py',
     'tests/test_refined_sampling.py','docs/REFINED_SAMPLING_PROTOCOL.md')
canonical=lambda x:json.loads(json.dumps(x,allow_nan=False))


def specs():
    variants=[('complete','封闭场景：全部条件',{}),('whole','保留 1 米整格',dict(divisions=1)),
              ('no-seed','起始空闲未知',dict(seed=False)),('no-upper','没有向上观察',dict(upper=False)),
              ('expired','起始声明提前过期',dict(expiry=10.)),('unknown-size','最小尺寸未知',dict(size=None)),
              ('obstacle','显式可见障碍',dict(obstacle=.2)),('open','开放场地缺回波',dict(enclosed=False)),
              ('thin','2 毫米反例',dict(obstacle=.002)),('two-edges','两段任务',dict(goal=[5,8]))]
    result=[]
    for key,title,changes in variants:
        s=dict(key=key,title=title,divisions=4,seed=True,expiry=12.,upper=True,size=.2,
               enclosed=True,obstacle=0.,goal=[4,8]);s.update(changes);result.append(s)
    return result


def world_for(s):
    surfaces=[Surface('ground',0,(-40,60,-40,56))]
    if s['enclosed']:
        surfaces.extend([Box('ceiling',3,(0,0,6),(20,16,6.2)),Box('west',3,(0,0,0),(.2,16,6)),
            Box('east',3,(19.8,0,0),(20,16,6)),Box('south',3,(0,0,0),(20,.2,6)),
            Box('north',3,(0,15.8,0),(20,16,6))])
    if s['obstacle']==.2:surfaces.append(Box('obstacle',4,(4.2,8.4,3.4),(4.4,8.6,3.6)))
    elif s['obstacle']:
        center=(4.2,8.507,3.507);half=s['obstacle']/2
        surfaces.append(Box('thin',4,tuple(x-half for x in center),tuple(x+half for x in center)))
    return World(tuple(surfaces))


class Vehicle(RefinedMissionVehicle):
    def __init__(self,s,saved=None):
        seed=InitialClearRegion((3.,8.,3.),(4.,9.,4.),0.,s['expiry'],'developer-declared-start') if s['seed'] else None
        assumption=None if s['size'] is None else SamplingAssumption(s['size'],s['size'])
        super().__init__(world_for(s),(3,8),divisions=s['divisions'],initial_region=seed,
                         upper_scan=s['upper'],sampling_assumption=assumption)
        self.saved=saved;self.saved_frames=[] if saved is None else saved['frames']
        self.frame_index=self.contact_index=0

    def capture(self,k,tick,aim=None):
        if self.saved is None:return super().capture(k,tick,aim)
        state=self.state();frame=RecordedPhysicalVehicle.capture(self,k,tick,aim)
        if self._scan_records is not None:self._scan_records.append((state,frame))
        return frame

    def contact_gap(self):
        value=super().contact_gap()
        if self.saved is not None:
            if self.contact_index>=len(self.saved['contact_readings']) or self.contact_readings[-1]!=self.saved['contact_readings'][self.contact_index]:
                raise ValueError('probe replay mismatch')
            self.contact_index+=1
        return value


def simulate(s,model,saved=None):
    v=Vehicle(s,saved)
    try:
        result=MissionSupervisor(v,model,(3,8),tuple(s['goal']),max_ticks=4).run()
        if v.history[-1]!=v.state():v.history.append(v.state())
        if saved is not None and (v.frame_index!=len(saved['frames']) or v.contact_index!=len(saved['contact_readings'])):
            raise ValueError('unused observations')
        raw=dict(frames=v.frames if saved is None else saved['frames'],commands=v.commands,contact_readings=v.contact_readings)
        summary=dict(spec=s,result=result,world=asdict(v.world),history=v.history,actual_final=v.state(),
            frames=len(raw['frames']),steps=v.steps,actions=v.actions,audit=v.audit,move_checks=v.move_checks,
            source_frames=v.source_frames,initial_region=None if v.sampling_history.initial_region is None else asdict(v.sampling_history.initial_region),
            dll_sha256=v.physics.dll_sha256,model_sha256=v.physics.model_sha256)
        return canonical(raw),canonical(summary)
    finally:v.close()


def page(report):
    rows=[];details=[]
    for c in report['cases']:
        s=c['summary'];r=s['result']
        rows.append('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in (c['title'],r['state'],
            len(s['actions']),s['audit']['clearance_violations'],r['original_reason'] or '模拟接地确认'))+'</tr>')
        guards=[]
        for q in s['move_checks']:
            g=q['final_guard'];sources=[p['source'] for row in g['sampling'] for p in row['patches']]
            guards.append(dict(state=q['state'],baseline=g['baseline'],boxes=len(sources),supported=sum(x is not None for x in sources),
                seed_boxes=sum(x is not None and x['kind']=='initial_assumption' for x in sources),
                camera_boxes=sum(x is not None and x['kind']=='camera' for x in sources),initial_region_active=g['initial_region_active']))
        details.append('<details><summary>'+escape(c['title'])+'：条件与覆盖</summary><pre>'+escape(json.dumps(dict(
            input=s['spec'],initial_region=s['initial_region'],moves=guards),ensure_ascii=False,indent=2))+'</pre></details>')
    return ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>近场细分与起始条件</title><style>
body{margin:0;background:#eef3f4;color:#203d48;font:16px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1150px;padding:28px 22px;margin:auto}p{line-height:1.8}h1{font-size:28px}.notice{padding:16px;border-left:4px solid #b77432;background:#fff1df}.table{overflow:auto}table{width:100%;background:white;border-collapse:collapse;font-size:13px}td,th{padding:12px;border-bottom:1px solid #d9e3e7;text-align:left}details{background:white;margin:12px 0;padding:18px;border-radius:8px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}a{color:#186477}</style>
<main><h1>近场空间细分：哪些条件让任务继续？</h1><p>每个 1 米格分成 64 个小盒，保留完整停止范围。不同小盒可由不同相机帧支持；近处盲区只能依赖明确给定、尚未过期的起始空闲声明。</p><p class="notice">封闭场地、向上扫描和起始空闲声明均为额外前提。相机没有回波仍记为未知；声明不会随无人机移动续期。点深度与尺寸假设不构成连续空间无障碍证明，表中失败和空间侵犯必须一起查看。</p><div class="table"><table><thead><tr><th>场景</th><th>任务终态</th><th>移动调用数</th><th>空间侵犯区间</th><th>原始原因</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>'''+''.join(details)+'''<p>移动调用不等于确认到达。接地也不等于配送完成；仍没有起飞、放餐、返航或真实视觉控制。</p><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定协议</a> · <a href="manifest.json">归档摘要</a></p></main></html>''').encode('utf-8')


def run(output,verify=False):
    check_manifest(PARENT);parent=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    hashes={n:sha(PROJECT/n) for n in sorted(set(parent['sources'])|set(OWN))}
    for n,d in parent['sources'].items():
        if hashes[n]!=d:raise ValueError('parent source changed')
    check_manifest(MODEL.parent)
    model=ColorModel.from_dict(json.loads(MODEL.read_text(encoding='utf-8')))
    if verify:
        count=check_manifest(output);report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if report['sources']!=hashes or report['parent_sha256']!=sha(PARENT/'report.json') or report['color_model_sha256']!=sha(MODEL):
            raise ValueError('source/input mismatch')
        if [c['summary']['spec'] for c in report['cases']]!=specs():raise ValueError('case matrix mismatch')
        for c in report['cases']:
            raw=json.loads(gzip.decompress((output/c['input']).read_bytes()))
            new,s=simulate(c['summary']['spec'],model,raw)
            if new!=raw or s!=c['summary']:raise ValueError('task replay mismatch: '+c['key'])
            with QuadrotorPhysics(model_xml=world_xml(saved_world(s['world']))) as physics:
                if replay_raw(physics,raw,(3,8))!=s['history'] or physics.state()!=s['actual_final']:
                    raise ValueError('independent motor replay mismatch')
        if page(report)!=(output/'demo.html').read_bytes():raise ValueError('page mismatch')
        return dict(verified=True,files=count,cases=len(report['cases']),frames=sum(c['summary']['frames'] for c in report['cases']),steps=sum(c['summary']['steps'] for c in report['cases']))
    if output.exists():raise FileExistsError('refuse overwrite '+str(output))
    output.mkdir(parents=True);cases=[]
    for spec in specs():
        started=perf_counter();raw,s=simulate(spec,model)
        name=spec['key']+'/raw.json.gz';put(output/name,gzip.compress(json.dumps(raw,separators=(',',':')).encode(),mtime=0))
        cases.append(dict(key=spec['key'],title=spec['title'],input=name,summary=s,wall_time_s=perf_counter()-started))
        print(spec['key'],s['result']['state'],len(s['actions']),s['audit']['clearance_violations'],flush=True)
    if hashes!={n:sha(PROJECT/n) for n in hashes}:raise ValueError('sources changed during experiment')
    report=dict(kind='refined-sampling-mission',sources=hashes,parent_sha256=sha(PARENT/'report.json'),color_model_sha256=sha(MODEL),cases=cases)
    dump(output/'report.json',report);put(output/'protocol.md',(PROJECT/'docs/REFINED_SAMPLING_PROTOCOL.md').read_bytes());put(output/'demo.html',page(report))
    dump(output/'manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases),frames=sum(c['summary']['frames'] for c in cases),steps=sum(c['summary']['steps'] for c in cases))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
