"""来源：本项目原创。实际 RGB-D 建图、组合动作检查及运动学对照。"""
import argparse
from dataclasses import asdict,replace
import gzip,json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.sampling_motion import SamplingAssumption,SamplingView,sampled_movement_guard
from drone_nav.pinhole import Intrinsics,Pose,PerspectiveFrame
from drone_nav.raycast import World,Surface,Box,render
from drone_nav.motion import MotionConfig,profile,simulate
from drone_nav.occupancy import VoxelMap
from drone_nav.localization import segment_box_clearance
from drone_nav.imaging import png_bytes
from drone_nav.realvision import sha,dump
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT=PROJECT/'work/sampling-coverage-01'
SOURCES=('drone_nav/sampling_motion.py','tools/sampling_motion_experiment.py','tests/test_sampling_motion.py',
         'drone_nav/sampling_motion_lab.html','drone_nav/motion.py','drone_nav/occupancy.py','drone_nav/localization.py',
         'drone_nav/raycast.py','drone_nav/pinhole.py','drone_nav/imaging.py','drone_nav/rendering.py')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
canonical=lambda v:json.loads(json.dumps(v,allow_nan=False))


def specs():
    cases=[]
    def add(key,title,**changes):
        s=dict(key=key,title=title,width=40,focal=25,position=[2.5,5.5,3.5],min_size=.2,
               now_s=1.,available_at_s=.1,captured_at_s=0.,clock_id='simulation-seconds',view_tick=0,
               margin=.15,error_bound=0.,hole=False,unknown=False,occupied=False,thin=False)
        s.update(changes);cases.append(s)
    add('unknown-assumption','最小特征尺寸未知',min_size=None)
    add('coarse-10cm','声明 10 厘米，巡航分辨率不足',min_size=.1)
    add('adequate-20cm','声明 20 厘米，必要门槛通过')
    add('higher-10cm','提高分辨率，同视场通过',width=160,focal=100,min_size=.1)
    add('higher-2cm','提高分辨率仍不足以支撑 2 厘米',width=160,focal=100,min_size=.02)
    add('long-stop-reserve','扩大停止预留，远端采样不足',margin=1.2)
    add('position-uncertainty','加入水平定位误差',error_bound=.2)
    add('cropped-view','仅放大焦距：完整格未进入画面',focal=100)
    add('current-camera-only','当前相机看不全包含自己的体素',position=[4.5,5.5,3.5])
    add('missing-pixel','检查窗口内有一个深度缺失',hole=True)
    add('wrong-clock','采样与地图时钟不一致',clock_id='other-clock')
    add('future-availability','观测尚未完成处理',available_at_s=2.)
    add('wrong-capture-time','采集秒与地图批次不一致',captured_at_s=.05)
    add('wrong-batch','观测批次错配',view_tick=1)
    add('expires-before-stop','预计停止前证据到期',now_s=10.)
    add('map-unknown','基础格被标记未知',unknown=True)
    add('map-occupied','基础格被标记占用',occupied=True)
    add('thin-correct-assumption','2 厘米漏采障碍：匹配尺寸假设时拒绝',thin=True,min_size=.02)
    add('thin-unknown-assumption','2 厘米漏采障碍：尺寸未知时拒绝',thin=True,min_size=None)
    add('thin-wrong-assumption','反例：2 厘米障碍被错误声明为至少 20 厘米',thin=True,min_size=.2)
    add('thin-visible','改变观察位置后采到细障碍',thin=True,position=[2.5,5.554,3.554])
    return cases


def world_for(spec):
    objects=[Surface('ground',0,(0,20,0,16)),Box('far-wall',3,(12,0,0),(12.2,16,8))]
    if spec['thin']:objects.append(Box('thin',4,(5.19,5.49,3.49),(5.21,5.51,3.51)))
    return World(tuple(objects))


def prepare(spec):
    w=spec['width'];h=w*3//4;k=Intrinsics(w,h,spec['focal'],spec['focal'],w/2-.5,h/2-.5)
    xyz=spec['position'];pose=Pose.look_at(tuple(xyz),(xyz[0]+1,xyz[1],xyz[2]))
    frame,truth=render(world_for(spec),pose,k,seed=901,tick=0)
    if spec['hole']:
        depth=list(frame.depth_z_m);depth[(h//2-1)*w+w//2-1]=None
        frame=replace(frame,depth_z_m=tuple(depth))
    return canonical(dict(spec=spec,frame=asdict(frame),thin_pixels=sum(o=='thin' for o in truth.objects)))


def audit_motion(world,motion,radius):
    if motion is None:return None
    points=[(4.5+s['distance_m'],5.5,3.5) for s in motion['samples']]
    distances=[segment_box_clearance(a,b,box) for a,b in zip(points,points[1:])
               for box in world.surfaces if isinstance(box,Box)]
    return dict(min_center_to_box_m=min(distances),collision=min(distances)<=radius,
                trajectory=[list(p) for p in points],scope='运动学轨迹线段与机体球/真值盒；仅事后评价')


def replay(raw):
    s=raw['spec'];f=raw['frame'];k=Intrinsics(**f['intrinsics']);pose=Pose(**f['pose'])
    frame=PerspectiveFrame(k,pose,f['rgb'],f['depth_z_m'],f['tick']);frame.validate()
    g=VoxelMap(20,16,8,100);g.integrate_moving([frame],0)
    # 这两项是显式地图输入故障；正常案例全部来自实际像素建图。
    if s['unknown']:g.free_seen.pop((5,5,3),None)
    if s['occupied']:g.occupied.add((5,5,3))
    view=SamplingView(replace(frame,tick=s['view_tick']),'historical-0','camera',s['clock_id'],
                      s['captured_at_s'],s['available_at_s'])
    config=MotionConfig(distance_margin_m=s['margin']);p=profile(1,config)
    assumption=None if s['min_size'] is None else SamplingAssumption(s['min_size'],s['min_size'])
    result=sampled_movement_guard(g,(4,5),(5,5),0,s['now_s'],{0:0.},p,config,views=[view],
                                 assumption=assumption,error_bound_m=s['error_bound'])
    before=simulate(p,config) if result['baseline']['allowed'] else None
    after=simulate(p,config) if result['allowed'] else None
    if result['allowed'] and not result['baseline']['allowed']:raise ValueError('sampling gate bypassed original rejection')
    states=[dict(voxel=v,state=g.state(tuple(v),0)) for v in result['required_cells']]
    world=world_for(s)
    return canonical(dict(guard=result,config=asdict(config),profile=asdict(p),voxel_states=states,
        baseline_motion=before,combined_motion=after,baseline_audit=audit_motion(world,before,config.body_radius_m),
        combined_audit=audit_motion(world,after,config.body_radius_m)))


def run(out,verify=False):
    if not verify and out.exists():raise FileExistsError('refuse overwrite '+str(out))
    check_manifest(PARENT);parent=read(PARENT/'report.json')
    for name,h in parent['sources'].items():
        if sha(PROJECT/name)!=h:raise ValueError('parent source changed: '+name)
    hashes={n:sha(PROJECT/n) for n in SOURCES}
    if verify:
        count=check_manifest(out);r=read(out/'report.json')
        if r['sources']!=hashes or r['parent_sha256']!=sha(PARENT/'report.json') or r['protocol_sha256']!=sha(out/'protocol.md'):
            raise ValueError('source or protocol changed')
        for c in r['cases']:
            raw=json.loads(gzip.decompress((out/c['input']).read_bytes()))
            if replay(raw)!=c['result'] or prepare(raw['spec'])!=raw:raise ValueError('replay or rerender differs '+c['key'])
            f=raw['frame'];k=f['intrinsics']
            if png_bytes(k['width'],k['height'],f['rgb'])!=(out/c['image']).read_bytes():raise ValueError('image differs')
        return dict(verified=True,files=count,cases=len(r['cases']),rerendered=len(r['cases']))
    out.mkdir(parents=True,exist_ok=False);cases=[]
    put(out/'protocol.md',(PROJECT/'docs/SAMPLING_MOTION_PROTOCOL.md').read_bytes())
    for spec in specs():
        raw=prepare(spec);result=replay(raw);key=spec['key'];input_name='inputs/'+key+'.json.gz';image_name='inputs/'+key+'.png'
        put(out/input_name,gzip.compress(json.dumps(raw,separators=(',',':'),allow_nan=False).encode('utf-8'),mtime=0))
        f=raw['frame'];put(out/image_name,png_bytes(spec['width'],spec['width']*3//4,f['rgb']))
        cases.append(dict(key=key,title=spec['title'],input=input_name,image=image_name,thin_pixels=raw['thin_pixels'],spec=spec,result=result))
    if hashes!={n:sha(PROJECT/n) for n in SOURCES}:raise ValueError('source changed')
    counts=dict(cases=len(cases),baseline_approved=sum(c['result']['guard']['baseline']['allowed'] for c in cases),
                combined_approved=sum(c['result']['guard']['allowed'] for c in cases),
                baseline_collisions=sum(bool(c['result']['baseline_audit'] and c['result']['baseline_audit']['collision']) for c in cases),
                combined_collisions=sum(bool(c['result']['combined_audit'] and c['result']['combined_audit']['collision']) for c in cases))
    r=dict(kind='sampling-and-motion-guard',sources=hashes,parent_sha256=sha(PARENT/'report.json'),
           protocol_sha256=sha(out/'protocol.md'),cases=cases,counts=counts,flight_authorized=False)
    dump(out/'report.json',r)
    html=(PROJECT/'drone_nav/sampling_motion_lab.html').read_text(encoding='utf-8')
    put(out/'demo.html',html.replace('__DATA__',json.dumps(r,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
