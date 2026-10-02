"""来源：本项目原创。固定主动观察方案的完整任务对照及物理重放。"""
import argparse
from dataclasses import asdict,replace
import gzip
from html import escape
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from drone_nav.active_observation import ActiveObservationMixin,ProbeConfig
from drone_nav.classifier import ColorModel
from drone_nav.mission_supervisor import MissionSupervisor
from drone_nav.pinhole import Pose
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.physical_vehicle import world_xml
from drone_nav.verify_coupled import saved_world
from drone_nav.verify_descent import replay_raw
from drone_nav.realvision import sha,dump
from tools.refined_sampling_experiment import Vehicle as RecordedVehicle,specs as old_specs,page as old_page
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT=PROJECT/'work/refined-sampling-01'
MODEL=PROJECT/'work/supervised-mission-01/model.json'
OWN=('drone_nav/active_observation.py','tools/active_observation_experiment.py',
     'tests/test_active_observation.py','docs/ACTIVE_OBSERVATION_PROTOCOL.md')
canonical=lambda x:json.loads(json.dumps(x,allow_nan=False))


def specs():
    base=next(s for s in old_specs() if s['key']=='two-edges')
    result=[]
    for key,title,probe,fault,delay in (
        ('off','原方案：关闭主动取景',{'enabled':False},None,0.),
        ('active','主动取景：两段任务',{},None,0.),
        ('sparse','较低采集频率',{'frame_interval_s':.08},None,0.),
        ('short','返回阶段时间不足',{'return_s':1.6},None,0.),
        ('delayed','等待消耗剩余期限',{},None,.4),
        ('depth','主动图像深度失效',{},'depth',0.),
        ('pose','主动图像位姿错配',{},'pose',0.)):
        result.append(dict(base,key=key,title=title,probe=probe,fault=fault,delay=delay))
    return result


class Vehicle(ActiveObservationMixin,RecordedVehicle):
    def __init__(self,s,saved=None):
        super().__init__(s,saved,probe_config=ProbeConfig(**s['probe']))
        self.spec=s;self.injected=0

    def observe_nearfield(self):
        if self.spec['delay']:self.hold(self.spec['delay'])
        return super().observe_nearfield()

    def capture(self,k,tick,aim=None):
        frame=super().capture(k,tick,aim)
        if self.probes and self.probes[-1]['status']=='OBSERVING':
            if self.spec['fault']=='depth':
                self.injected+=1
                return replace(frame,depth_z_m=(None,)*(k.width*k.height))
            if self.spec['fault']=='pose' and len(self.probes[-1]['frames'])==1:
                self.injected+=1;p=frame.pose
                return replace(frame,pose=Pose((p.position[0]+.1,*p.position[1:]),p.right,p.down,p.forward))
        return frame


def simulate(s,model,saved=None):
    v=Vehicle(s,saved)
    try:
        result=MissionSupervisor(v,model,(3,8),(5,8),max_ticks=4).run()
        if v.history[-1]!=v.state():v.history.append(v.state())
        if saved is not None and (v.frame_index!=len(saved['frames']) or v.contact_index!=len(saved['contact_readings'])):
            raise ValueError('unconsumed observations')
        raw=dict(frames=v.frames if saved is None else saved['frames'],commands=v.commands,contact_readings=v.contact_readings)
        summary=dict(spec=s,result=result,world=asdict(v.world),history=v.history,actual_final=v.state(),
            frames=len(raw['frames']),steps=v.steps,actions=v.actions,audit=v.audit,move_checks=v.move_checks,
            source_frames=v.source_frames,probes=v.probes,fault_injections=v.injected,
            initial_region=asdict(v.sampling_history.initial_region),dll_sha256=v.physics.dll_sha256,
            model_sha256=v.physics.model_sha256)
        return canonical(raw),canonical(summary)
    finally:v.close()


def page(report):
    base=old_page(report).decode('utf-8').replace('近场细分与起始条件','主动取景与多段任务').replace(
        '近场空间细分：哪些条件让任务继续？','主动取景：让第二段取得新证据')
    details=[]
    for c in report['cases']:
        entries=[{k:p.get(k) for k in ('status','support_until_s','actual_start','actual_final','stable_duration_s',
                    'max_center_error_m','max_speed_mps')}|{'new_frames':len(p['frames'])} for p in c['summary']['probes']]
        details.append('<details><summary>'+escape(c['title'])+'：实际取景与返回</summary><pre>'+escape(json.dumps(entries,ensure_ascii=False,indent=2))+'</pre></details>')
    note='<p class="notice">主动取景使用原控制器小幅移动机体，再返回航点；只在上一段条件支持的范围和期限内执行。新的图像保留实际时间，不刷新初始声明。仍是封闭场景开发对照，旧细障碍反例没有被解决。</p>'
    return base.replace('<main>','<main>'+note,1).replace('</main>',''.join(details)+'</main>').encode('utf-8')


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
        print(spec['key'],s['result']['state'],s['result']['original_reason'],len(s['actions']),flush=True)
    if hashes!={n:sha(PROJECT/n) for n in hashes}:raise ValueError('source changed during run')
    report=dict(kind='active-observation',sources=hashes,parent_sha256=sha(PARENT/'report.json'),color_model_sha256=sha(MODEL),cases=cases)
    dump(output/'report.json',report);put(output/'protocol.md',(PROJECT/'docs/ACTIVE_OBSERVATION_PROTOCOL.md').read_bytes());put(output/'demo.html',page(report))
    dump(output/'manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases),frames=sum(c['summary']['frames'] for c in cases),steps=sum(c['summary']['steps'] for c in cases))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
