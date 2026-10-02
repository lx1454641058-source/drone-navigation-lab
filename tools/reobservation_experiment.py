"""来源：本项目原创。解析像素角域下界、重观测解除与路线对照。"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import product
import json
from math import ceil, floor
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.detection_bridge import DetectionPacket, SpatialContext
from drone_nav.pinhole import Intrinsics, Pose
from drone_nav.observation_channel import SurfaceEvidenceChannel
from drone_nav.occupancy import VoxelMap
from drone_nav.reobservation import FOOTPRINT_MIN, ClearanceConfig, coverage_window, reobserved_route
from drone_nav.realvision import sha, dump
from tools.channel_experiment import load_input
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT=PROJECT/'work/observation-channel-02'
SOURCES=('drone_nav/reobservation.py','tools/reobservation_experiment.py','drone_nav/reobservation_lab.html',
         'tests/test_reobservation.py','drone_nav/observation_channel.py','drone_nav/detection_bridge.py',
         'drone_nav/pinhole.py','drone_nav/occupancy.py','drone_nav/planning.py','tools/channel_experiment.py')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
canonical=lambda v:json.loads(json.dumps(v,allow_nan=False))


def footprint_depth(k, pose, boxes, far_z=10.):
    """包围盒投影矩形覆盖整个角域，最近角深度是最近表面的下界。

    允许假遮挡；不将中心射线未命中解释为整像素角域无障碍。
    盒跨过相机平面时不做裁剪猜测，而是拒绝整帧。
    """
    depth=[far_z]*(k.width*k.height)
    for box in boxes:
        corners=[pose.project(v,k) for v in product(*zip(box[:3],box[3:]))]
        if any(v is None for v in corners):
            return [None]*len(depth)
        x0=max(0,ceil(min(v[0] for v in corners)-.5-1e-9))
        x1=min(k.width-1,floor(max(v[0] for v in corners)+.5+1e-9))
        y0=max(0,ceil(min(v[1] for v in corners)-.5-1e-9))
        y1=min(k.height-1,floor(max(v[1] for v in corners)+.5+1e-9))
        nearest=min(v[2] for v in corners)
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):depth[y*k.width+x]=min(depth[y*k.width+x],nearest)
    return depth


def frame(boxes=(), t=2.):
    k=Intrinsics(96,72,50,50,47.5,35.5)
    pose=Pose.look_at((10,8,1.5),(15,8,1.5))
    p=DetectionPacket('geometry-new','geometry-camera',96,72,(),'analytic_geometry',t,t+.05,'synthetic-seconds')
    c=SpatialContext(p.frame_id,p.camera_id,p.clock_id,t,t,k,pose,tuple(footprint_depth(k,pose,boxes)),
                     True,'camera_optical_axis_z_m','analytic_footprint_lower_bound','exact_fixture_pose','fixture-k50')
    return dict(packet=asdict(p),context=asdict(c))


def specs():
    cases=[]
    def add(key,title,boxes=(),**changes):
        item=dict(key=key,title=title,geometry=frame(boxes),boxes=list(boxes),sampling_model=FOOTPRINT_MIN,
                  now_s=2.1,free_rows=[6,7,8],occupied=[],tick=0,new_positive=False)
        item.update(changes);cases.append(item);return item
    add('clear-wide','障碍移走：宽走廊恢复直行')
    add('clear-narrow','障碍移走：单通道恢复路线',free_rows=[7])
    add('still-occupied','原位置仍有障碍',boxes=[[14.2,7.1,.1,14.4,8.9,1.9]],free_rows=[7])
    add('foreground','前景遮挡：不能看穿遮挡解除',boxes=[[12,6,-1,12.2,10,4]],free_rows=[7])
    thin=[[14.51,7.991,1.21,14.52,7.993,1.22]]
    add('thin-obstacle','毫米级窄障碍：角域下界保留限制',boxes=thin,free_rows=[7])
    item=add('point-samples-thin','中心射线可能漏掉细障碍：拒绝该输入契约',boxes=thin,
             sampling_model='point_samples',free_rows=[7])
    item['geometry']=frame()  # 模拟未命中细障碍的中心点读数，绝不冒充角域下界。
    item=add('missing-depth','投影边缘一个像素缺失',free_rows=[7])
    p,c=load_input(item['geometry']);w=coverage_window([14,7,1],c,ClearanceConfig())['window']
    item['geometry']['context']['depth_z_m']=list(c.depth_z_m)
    item['geometry']['context']['depth_z_m'][w[1]*96+w[0]]=None
    item=add('partial-view','放大视角：整个体素未进入画面',free_rows=[7])
    item['geometry']['context']['intrinsics'].update(fx=300,fy=300)
    item=add('behind-camera','目标格位于相机后方',free_rows=[7])
    item['geometry']['context']['pose']=asdict(Pose.look_at((10,8,1.5),(5,8,1.5)))
    add('expiry-edge','重复消费至有效期边界',now_s=2.5,free_rows=[7])
    add('expired','同一解除帧到期：恢复未知限制',now_s=2.51,free_rows=[7])
    item=add('wrong-frame','深度帧身份错配',free_rows=[7]);item['geometry']['context']['frame_id']='wrong'
    item=add('wrong-clock','跨时钟证据被拒绝',free_rows=[7])
    item['geometry']['packet']['clock_id']='other';item['geometry']['context']['clock_id']='other'
    add('older','较早采集的空闲图不能推翻旧表面',geometry=frame(t=.95),now_s=1.2,free_rows=[7])
    add('sync-overlap','时间区间与旧表面重叠',geometry=frame(t=1.01),now_s=1.2,free_rows=[7])
    item=add('unsynchronized','位姿与深度错时',free_rows=[7]);item['geometry']['context']['pose_at_s']=2.05
    item=add('depth-boundary','深度减去误差刚好等于最远边界',free_rows=[7])
    item['geometry']['context']['depth_z_m']=[5.15]*(96*72)
    item=add('depth-range','深度超出可信量程',free_rows=[7])
    item['geometry']['context']['depth_z_m']=[100.]*(96*72)
    add('base-unknown','基础地图未知：解除附加限制仍不能通行',free_rows=[])
    add('base-occupied','基础地图占用：解除附加限制仍不能通行',free_rows=[7],occupied=[[14,7,1]])
    add('base-expired','基础空闲证据到期：不能借解除创建空闲',free_rows=[7],tick=101)
    add('new-positive','较新正向表面重新阻塞',free_rows=[7],new_positive=True,now_s=2.3)
    return cases


def simulate(seed,spec):
    ch=SurfaceEvidenceChannel(20,16,8,clock_id='synthetic-seconds')
    p,c=load_input(seed);ch.ingest(p,c,now_s=1.2)
    if spec['new_positive']:
        p=replace(p,frame_id='later-positive',captured_at_s=2.1,completed_at_s=2.2)
        c=replace(c,frame_id=p.frame_id,depth_at_s=2.1,pose_at_s=2.1)
        ch.ingest(p,c,now_s=2.2)
    g=VoxelMap(20,16,8,ttl_ticks=100)
    g.free_seen={(x,y,1):0 for x in range(11,18) for y in spec['free_rows']}
    g.occupied={tuple(c) for c in spec['occupied']}
    packet,context=load_input(spec['geometry'])
    result=reobserved_route(g,ch,packet,context,sampling_model=spec['sampling_model'],
        layer=1,tick=spec['tick'],now_s=spec['now_s'],start=(11,7),goal=(17,7))
    forbidden={tuple(v) for v in result['before']['base_blocked']+result['restrictions']}
    if any(tuple(v) in forbidden for v in result['route']):raise ValueError('route crosses blocked cell')
    if result['flight_authorized']:raise ValueError('flight is not authorized')
    # 评价独立读取真值盒；决策模块从未接收 boxes。
    evaluation=[]
    for cell in result['clearance']['cells']:
        v=cell['voxel'];margin=.1
        obstructed=any(all(v[i]-margin <= box[i+3] and v[i]+1+margin >= box[i] for i in range(3))
                       for box in spec['boxes'])
        evaluation.append(dict(voxel=v,truth_obstructed=obstructed,false_clear=obstructed and cell['cleared'],
                               free_but_retained=not obstructed and not cell['cleared']))
    return canonical(dict(decision=result,evaluation=evaluation))


def run(out,verify=False):
    if not verify and out.exists():raise FileExistsError('refuse overwrite '+str(out))
    check_manifest(PARENT);parent=read(PARENT/'report.json')
    for name,h in parent['sources'].items():
        if sha(PROJECT/name)!=h:raise ValueError('parent source changed: '+name)
    hashes={name:sha(PROJECT/name) for name in SOURCES}
    if verify:
        count=check_manifest(out);r=read(out/'report.json');seed=read(out/'seed.json')
        if r['sources']!=hashes or r['parent_sha256']!=sha(PARENT/'report.json') or r['protocol_sha256']!=sha(out/'protocol.md'):
            raise ValueError('source or protocol changed')
        for c in r['cases']:
            if simulate(seed,read(out/c['input']))!=c['result']:raise ValueError('replay differs '+c['key'])
        return dict(verified=True,files=count,cases=len(r['cases']))
    out.mkdir(parents=True,exist_ok=False)
    seed=read(PARENT/'wide/input.json')['events'][0]['input'];dump(out/'seed.json',seed)
    put(out/'protocol.md',(PROJECT/'docs/REOBSERVATION_PROTOCOL.md').read_bytes())
    cases=[]
    for spec in specs():
        result=simulate(seed,spec);relative='inputs/'+spec['key']+'.json'
        put(out/relative,(json.dumps(spec,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf-8'))
        cases.append(dict(key=spec['key'],title=spec['title'],input=relative,now_s=spec['now_s'],
                          depth=spec['geometry']['context']['depth_z_m'],result=result))
    if hashes!={n:sha(PROJECT/n) for n in SOURCES}:raise ValueError('source changed during experiment')
    counts=dict(cases=len(cases),voxel_assessments=sum(len(c['result']['evaluation']) for c in cases),
                false_clear=sum(e['false_clear'] for c in cases for e in c['result']['evaluation']),
                free_but_retained=sum(e['free_but_retained'] for c in cases for e in c['result']['evaluation']))
    r=dict(kind='conditional-geometric-clearance',sources=hashes,parent_sha256=sha(PARENT/'report.json'),
           protocol_sha256=sha(out/'protocol.md'),counts=counts,cases=cases)
    dump(out/'report.json',r)
    html=(PROJECT/'drone_nav/reobservation_lab.html').read_text(encoding='utf-8')
    put(out/'demo.html',html.replace('__DATA__',json.dumps(r,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
