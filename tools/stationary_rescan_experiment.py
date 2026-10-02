"""来源：本项目原创。原地相位复查；世界只在传感器与评价侧可见。"""
import argparse,gzip,json,sys
from dataclasses import asdict,replace
from pathlib import Path
PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.stationary_rescan import SamplingHistory,verify_stationary
from drone_nav.sampling_motion import SamplingView,SamplingAssumption
from drone_nav.pinhole import Intrinsics,Pose,PerspectiveFrame
from drone_nav.raycast import World,Surface,Box,render
from drone_nav.motion import MotionConfig,profile,simulate
from drone_nav.imaging import png_bytes
from drone_nav.realvision import sha,dump
from tools.sampling_motion_experiment import audit_motion
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT=PROJECT/'work/sampling-motion-01'
SOURCES=('drone_nav/stationary_rescan.py','tools/stationary_rescan_experiment.py','tests/test_stationary_rescan.py',
         'drone_nav/stationary_rescan_lab.html','drone_nav/sampling_motion.py','drone_nav/occupancy.py',
         'drone_nav/motion.py','drone_nav/pinhole.py','drone_nav/raycast.py','drone_nav/localization.py',
         'tools/sampling_motion_experiment.py','drone_nav/imaging.py','drone_nav/rendering.py')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
canonical=lambda v:json.loads(json.dumps(v,allow_nan=False))


def specs():
    cases=[]
    def add(key,title,**kwargs):
        s=dict(key=key,title=title,width=40,min_size=.2,thin_size=0.,center=[5.2,5.5,3.5],
               history=True,now_s=1.,settle_s=.1,fault=None);s.update(kwargs);cases.append(s)
    add('empty','空场景与有效历史')
    add('hidden-20mm','隐藏 2 厘米障碍，错误 20 厘米声明',thin_size=.02)
    add('hidden-unknown','隐藏 2 厘米障碍，尺寸未知',thin_size=.02,min_size=None)
    add('hidden-2mm-offset','偏移 2 毫米障碍，仍可能漏采',thin_size=.002,center=[5.2,5.507,3.507])
    add('empty-unknown','空场景，尺寸声明未知',min_size=None)
    add('coarse','低分辨率与 10 厘米声明',min_size=.1)
    add('high-empty','高分辨率空场景',width=160,min_size=.1)
    add('high-hidden','高分辨率隐藏 2 厘米',width=160,min_size=.1,thin_size=.02)
    add('expired-history','历史完整覆盖已到期',now_s=10.)
    add('no-history','没有历史，保留近场盲区',history=False)
    add('depth-loss','第二帧全部深度丢失',fault='depth')
    add('clock-fault','第二帧时钟错配',fault='clock')
    add('slow-scan','慢扫描消耗历史有效期',settle_s=2.95)
    return cases


def world_for(spec):
    items=[Surface('ground',0,(0,20,0,16)),Box('wall',3,(12,0,0),(12.2,16,8))]
    if spec['thin_size']:
        half=spec['thin_size']/2;c=spec['center']
        items.append(Box('thin',4,tuple(x-half for x in c),tuple(x+half for x in c)))
    return World(tuple(items))


def pack(view,hits):return canonical(dict(view=asdict(view),thin_pixels=hits))


def unpack(raw):
    d=raw['view'];f=d['frame'];pose=Pose(**{k:tuple(v) for k,v in f['pose'].items()})
    frame=PerspectiveFrame(Intrinsics(**f['intrinsics']),pose,f['rgb'],f['depth_z_m'],f['tick'])
    return SamplingView(**dict(d,frame=frame))


class Source:
    def __init__(self,spec,saved=None):self.spec=spec;self.saved=saved;self.records=[]
    def capture(self,pose,k,tick,captured,available):
        index=len(self.records)
        if self.saved is not None:
            if index>=len(self.saved):raise ValueError('replay needs unavailable frame')
            raw=self.saved[index];view=unpack(raw)
        else:
            frame,truth=render(world_for(self.spec),pose,k,tick=tick,seed=901)
            if index==1 and self.spec['fault']=='depth':frame=replace(frame,depth_z_m=(None,)*(k.width*k.height))
            clock='other' if index==1 and self.spec['fault']=='clock' else 'simulation-seconds'
            view=SamplingView(frame,'rescan-'+str(index),'camera',clock,captured,available)
            raw=pack(view,sum(v=='thin' for v in truth.objects))
        self.records.append(raw);return view


def simulate_case(spec,mode,saved=None):
    h=SamplingHistory();w=spec['width'];k=Intrinsics(w,w*3//4,25*w/40,25*w/40,w/2-.5,w*3/8-.5)
    historical=None
    if spec['history']:
        if saved is None:
            f,t=render(world_for(spec),Pose.look_at((2.5,5.5,3.5),(3.5,5.5,3.5)),k,tick=0,seed=901)
            historical=pack(SamplingView(f,'historical','camera','simulation-seconds',0.,.1),sum(v=='thin' for v in t.objects))
        else:historical=saved['historical']
        h.add(unpack(historical),now_s=.1)
    source=Source(spec,None if saved is None else saved['scans']);config=MotionConfig()
    assumption=None if spec['min_size'] is None else SamplingAssumption(spec['min_size'],spec['min_size'])
    result=verify_stationary(h,source,(4,5),(5,5),k,config=config,assumption=assumption,
                             mode=mode,now_s=spec['now_s'],settle_s=spec['settle_s'])
    if saved is not None and len(source.records)!=len(saved['scans']):raise ValueError('unused replay frames')
    initial_motion=simulate(profile(1,config),config) if result['initial']['allowed'] else None
    result.update(initial_audit=audit_motion(world_for(spec),initial_motion,config.body_radius_m),
                  final_audit=audit_motion(world_for(spec),result['motion'],config.body_radius_m))
    return canonical(dict(spec=spec,mode=mode,historical=historical,scans=source.records)),canonical(result)


def run(out,verify=False):
    if not verify and out.exists():raise FileExistsError('refuse overwrite '+str(out))
    check_manifest(PARENT);parent=read(PARENT/'report.json')
    for n,h in parent['sources'].items():
        if sha(PROJECT/n)!=h:raise ValueError('parent source changed: '+n)
    hashes={n:sha(PROJECT/n) for n in SOURCES}
    if verify:
        count=check_manifest(out);r=read(out/'report.json');frames=0
        if r['sources']!=hashes or r['parent_sha256']!=sha(PARENT/'report.json') or r['protocol_sha256']!=sha(out/'protocol.md'):
            raise ValueError('source or protocol differs')
        for c in r['cases']:
            raw=json.loads(gzip.decompress((out/c['input']).read_bytes()))
            replayed,result=simulate_case(raw['spec'],raw['mode'],raw)
            fresh,fresh_result=simulate_case(raw['spec'],raw['mode'])
            if replayed!=raw or fresh!=raw or result!=c['result'] or fresh_result!=result:raise ValueError('replay/rerender differs')
            for image,frame in zip(c['images'],raw['scans']):
                f=frame['view']['frame'];k=f['intrinsics']
                if png_bytes(k['width'],k['height'],f['rgb'])!=(out/image).read_bytes():raise ValueError('PNG differs')
            frames+=len(raw['scans'])+bool(raw['historical'])
        return dict(verified=True,files=count,runs=len(r['cases']),rerendered_frames=frames)
    out.mkdir(parents=True,exist_ok=False);cases=[]
    put(out/'protocol.md',(PROJECT/'docs/STATIONARY_RESCAN_PROTOCOL.md').read_bytes())
    for spec in specs():
        for mode in ('repeat','phase_diverse'):
            raw,result=simulate_case(spec,mode);key=spec['key']+'-'+mode;name=key+'/input.json.gz'
            put(out/name,gzip.compress(json.dumps(raw,separators=(',',':'),allow_nan=False).encode('utf-8'),mtime=0));images=[]
            for i,frame in enumerate(raw['scans']):
                f=frame['view']['frame'];k=f['intrinsics'];img=key+f'/scan-{i}.png'
                put(out/img,png_bytes(k['width'],k['height'],f['rgb']));images.append(img)
            cases.append(dict(key=key,title=spec['title'],mode=mode,input=name,images=images,
                              thin_pixels=[v['thin_pixels'] for v in raw['scans']],result=result))
    if hashes!={n:sha(PROJECT/n) for n in SOURCES}:raise ValueError('source changed')
    r=dict(kind='stationary-phase-rescan',sources=hashes,parent_sha256=sha(PARENT/'report.json'),
           protocol_sha256=sha(out/'protocol.md'),cases=cases)
    dump(out/'report.json',r)
    html=(PROJECT/'drone_nav/stationary_rescan_lab.html').read_text(encoding='utf-8')
    put(out/'demo.html',html.replace('__DATA__',json.dumps(r,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return dict(runs=len(cases),scans=sum(len(c['images']) for c in cases),
        moves=sum(c['result']['motion'] is not None for c in cases),
        collisions=sum(bool(c['result']['final_audit'] and c['result']['final_audit']['collision']) for c in cases))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
