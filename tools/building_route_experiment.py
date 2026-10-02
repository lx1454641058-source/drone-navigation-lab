"""来源：本项目原创。建筑遮挡路线与路段故障的连续物理任务。"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import argparse
import gzip
from html import escape
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.active_observation import ActiveObservationMixin,ProbeConfig
from drone_nav.refined_sampling import RefinedMissionVehicle,InitialClearRegion
from drone_nav.continuation_history import ContinuationHistory
from drone_nav.sampling_motion import SamplingAssumption
from drone_nav.raycast import World,Surface,Box
from drone_nav.mission_supervisor import MissionSupervisor
from drone_nav.wide_scan import WideScanMixin
from drone_nav.mission_goal_probe import MissionGoalProbeMixin
from drone_nav.classifier import ColorModel
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.physical_vehicle import world_xml
from drone_nav.verify_coupled import saved_world
from drone_nav.verify_descent import replay_raw
from drone_nav.pinhole import Pose,Intrinsics
from drone_nav.range_observation import render_range
from drone_nav.realvision import sha,dump
from tools.range_observation_experiment import Vehicle as RangeVehicle,canonical,MODEL
from tools.route_continuation_experiment import page as route_page
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT=ROOT/'work/route-continuation-01'
OWN=('drone_nav/wide_scan.py','drone_nav/mission_goal_probe.py','tools/building_route_experiment.py',
     'tests/test_wide_scan.py','docs/BUILDING_ROUTE_PROTOCOL.md')


def world_for(s):
    return World((Surface('ground',0,(-40,60,-40,56)),
                  *(Box(b['name'],3,tuple(b['low']),tuple(b['high'])) for b in s['buildings'])))


def base_spec():
    return dict(key='building',title='建筑阻挡直达方向',goal=[9,8],max_ticks=24,
        divisions=4,expiry=12.,size=.2,upper=True,probe={},explicit=True,range_m=30.,
        range_fault=None,sensor_after_moves=None,ignore_move=None,wide_scan=False,level_scan=False,mission_goal_probe=False,
        buildings=[dict(name='building',low=[5.2,7.2,0.],high=[6.8,9.8,5.])])


def specs():
    base=base_spec()
    return [dict(base,key=k,title=t,**changes) for k,t,changes in (
        ('narrow','原视野：建筑绕行',{}),
        ('wide-down','扩大视野，仍向下扫描',dict(wide_scan=True)),
        ('level','水平环视，保留旧终点判断',dict(wide_scan=True,level_scan=True)),
        ('mission-goal','区分观察点与最终任务目标',dict(wide_scan=True,level_scan=True,mission_goal_probe=True)),
        ('depth-stage','第一段后测距持续失效',dict(wide_scan=True,level_scan=True,mission_goal_probe=True,sensor_after_moves=1)),
        ('ignored-stage','第二个移动航点未执行',dict(wide_scan=True,level_scan=True,mission_goal_probe=True,ignore_move=1)))]


class Vehicle(WideScanMixin,MissionGoalProbeMixin,ActiveObservationMixin,RefinedMissionVehicle):
    def __init__(self,s,saved=None):
        if type(s['wide_scan']) is not bool:raise ValueError('wide_scan must be explicit bool')
        if type(s['level_scan']) is not bool or (s['level_scan'] and not s['wide_scan']):
            raise ValueError('level_scan requires explicit wide scan')
        if type(s['mission_goal_probe']) is not bool:raise ValueError('mission_goal_probe must be bool')
        goal=s['goal']
        if (not isinstance(goal,(tuple,list)) or len(goal)!=2 or any(type(x) is not int for x in goal)
                or not 1<=goal[0]<19 or not 1<=goal[1]<15):raise ValueError('interior mission goal required')
        threshold=s['sensor_after_moves']
        if threshold is not None and (type(threshold) is not int or threshold<1):
            raise ValueError('sensor stage must be a positive move count')
        ignored=s['ignore_move']
        if ignored is not None and (type(ignored) is not int or ignored<0):
            raise ValueError('ignored move must be a nonnegative action index')
        seed=InitialClearRegion((3.,8.,3.),(4.,9.,4.),0.,s['expiry'],'developer-declared-start')
        super().__init__(world_for(s),(3,8),divisions=s['divisions'],initial_region=seed,
            upper_scan=s['upper'],sampling_assumption=SamplingAssumption(s['size'],s['size']),
            probe_config=ProbeConfig(**s['probe']),ignore_move=ignored)
        self.sampling_history=ContinuationHistory(divisions=s['divisions'],initial_region=seed,
            horizon_s=self.budget.max_move_s+.5,free_ttl_s=self.budget.free_ttl_s)
        self.spec=s;self.saved=saved;self.saved_frames=[] if saved is None else saved['frames']
        self.frame_index=self.contact_index=0;self.fault_frames=0

    def capture(self,k,tick,aim=None):
        original=self.spec;threshold=original['sensor_after_moves']
        active=threshold is not None and len(self.actions)>=threshold
        # 触发依据是已执行动作数，不读取障碍真值或预定成功路线。
        if self.saved is not None:
            if self.frame_index>=len(self.saved_frames) or self.saved_frames[self.frame_index]['sensor_fault_active']!=active:
                raise ValueError('sensor fault stage replay mismatch')
        self.spec=dict(original,range_fault='all' if active else None)
        try:frame=RangeVehicle.capture(self,k,tick,aim)
        finally:self.spec=original
        if self.saved is None:self.frames[-1]['sensor_fault_active']=active
        self.fault_frames+=int(active)
        return frame

    def contact_gap(self):
        value=super().contact_gap()
        if self.saved is not None:
            if (self.contact_index>=len(self.saved['contact_readings']) or
                    self.contact_readings[-1]!=self.saved['contact_readings'][self.contact_index]):
                raise ValueError('contact replay mismatch')
            self.contact_index+=1
        return value


def simulate(s,model,saved=None):
    v=Vehicle(s,saved)
    try:
        result=MissionSupervisor(v,model,(3,8),tuple(s['goal']),max_ticks=s['max_ticks']).run()
        if v.history[-1]!=v.state():v.history.append(v.state())
        frames=v.frames if saved is None else saved['frames']
        if saved is not None and (v.frame_index!=len(frames) or v.contact_index!=len(saved['contact_readings'])):
            raise ValueError('unused building observations')
        raw=dict(frames=frames,commands=v.commands,contact_readings=v.contact_readings)
        h=v.sampling_history
        summary=dict(spec=s,result=result,world=asdict(v.world),history=v.history,actual_final=v.state(),
            frames=len(frames),steps=v.steps,actions=v.actions,audit=v.audit,move_checks=v.move_checks,
            source_frames=v.source_frames,probes=v.probes,fault_frames=v.fault_frames,
            range_outcomes=dict(Counter(x for item in frames for x in item['frame']['outcomes'])),
            initial_region=asdict(h.initial_region),dll_sha256=v.physics.dll_sha256,model_sha256=v.physics.model_sha256,
            cache=dict(retired=h.retired,peak_frames=h.peak_frames,final_frames=len(h._entries)))
        return canonical(raw),canonical(summary)
    finally:v.close()


def page(report):
    html=route_page(report).decode('utf-8')
    html=html.replace('长路线与转弯验证','建筑遮挡与途中异常').replace('从两段直线到连续路线与转弯','建筑挡住直达方向后，如何继续？')
    html=html.replace('原方法在第二次主动补拍时达到 128 帧上限。新方法只移除已无法覆盖下一段最早停止期限的图像，保留地图障碍、原时间与全部输入归档。',
        '相同建筑场景对照原视野、扩大视野与水平环视；不修改停止范围、控制预算或来源有效期。规划器只读观测，建筑真值仅用于传感器、物理计算及事后评价。')
    html=html.replace('静态无障碍场地','静态建筑场地').replace('绿点为起点，橙圈为目标；','灰色块为建筑，绿点为起点，橙圈为目标；')
    for case in report['cases']:
        marker='<svg viewBox="0 0 560 448" role="img" aria-label="'+escape(case['title'])+'的俯视轨迹"><rect x="0" y="0" width="560" height="448" fill="#f2f6f7"/>'
        shapes=[]
        for b in case['summary']['spec']['buildings']:
            a,z=b['low'],b['high']
            shapes.append(f'<rect x="{a[0]*28}" y="{(16-z[1])*28}" width="{(z[0]-a[0])*28}" height="{(z[1]-a[1])*28}" fill="#9aabb0"/>')
        html=html.replace(marker,marker+''.join(shapes))
    return html.replace('viewBox="0 0 560 448"','viewBox="56 112 280 252"').encode('utf-8')


def archive_case(spec,model,output):
    raw,s=simulate(spec,model);name=spec['key']+'/raw.json.gz'
    put(output/name,gzip.compress(json.dumps(raw,separators=(',',':')).encode(),mtime=0))
    return dict(key=spec['key'],title=spec['title'],input=name,summary=s)


def verify_case(case,model,output):
    raw=json.loads(gzip.decompress((output/case['input']).read_bytes()))
    new,s=simulate(case['summary']['spec'],model,raw)
    if new!=raw or s!=case['summary']:raise ValueError('building replay differs '+case['key'])
    world=saved_world(s['world'])
    with QuadrotorPhysics(model_xml=world_xml(world)) as physics:
        if replay_raw(physics,raw,(3,8))!=s['history'] or physics.state()!=s['actual_final']:raise ValueError('building motor replay differs')
    for item in raw['frames']:
        d=item['frame'];p=Pose(**{k:tuple(v) for k,v in d['pose'].items()})
        f=render_range(world,p,Intrinsics(**d['intrinsics']),tick=d['tick'],max_range_m=s['spec']['range_m'],invalid=item['sensor_fault_active'])
        if canonical(asdict(f))!=d:raise ValueError('building rerender differs')
    return case['key']


def run(output,verify=False):
    check_manifest(PARENT);parent=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    hashes={n:sha(ROOT/n) for n in sorted(set(parent['sources'])|set(OWN))}
    if any(hashes[n]!=d for n,d in parent['sources'].items()):raise ValueError('parent source changed')
    check_manifest(MODEL.parent);model=ColorModel.from_dict(json.loads(MODEL.read_text(encoding='utf-8')))
    if verify:
        files=check_manifest(output);r=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if r['sources']!=hashes or r['parent_sha256']!=sha(PARENT/'report.json'):raise ValueError('source/input mismatch')
        if [c['summary']['spec'] for c in r['cases']]!=specs():raise ValueError('matrix mismatch')
        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(verify_case,c,model,output) for c in r['cases']]
            for j in as_completed(jobs):print('verified',j.result(),flush=True)
        if page(r)!=(output/'demo.html').read_bytes():raise ValueError('page mismatch')
        return dict(verified=True,files=files,cases=len(r['cases']))
    if output.exists():raise FileExistsError('refuse overwrite '+str(output))
    output.mkdir(parents=True);found={}
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(archive_case,s,model,output) for s in specs()]
        for j in as_completed(jobs):
            c=j.result();found[c['key']]=c;s=c['summary']
            print(c['key'],s['result']['state'],s['result']['original_reason'],s['result']['confirmed_cell'],flush=True)
    cases=[found[s['key']] for s in specs()]
    if hashes!={n:sha(ROOT/n) for n in hashes}:raise ValueError('source changed during run')
    r=dict(kind='building-route',sources=hashes,parent_sha256=sha(PARENT/'report.json'),cases=cases)
    dump(output/'report.json',r);put(output/'demo.html',page(r));put(output/'protocol.md',(ROOT/'docs/BUILDING_ROUTE_PROTOCOL.md').read_bytes())
    dump(output/'manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases),frames=sum(c['summary']['frames'] for c in cases),steps=sum(c['summary']['steps'] for c in cases))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
