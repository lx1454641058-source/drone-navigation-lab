"""来源：本项目原创。观测事件缓存与受控走廊的影子路线回放。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.detection_bridge import DetectionPacket,SpatialContext
from drone_nav.pinhole import Intrinsics,Pose
from drone_nav.observation_channel import SurfaceEvidenceChannel,shadow_route
from drone_nav.occupancy import VoxelMap
from drone_nav.realvision import sha,dump
from tools.setup_vision import put
from tools.object_probe import check_manifest

SOURCE=PROJECT/'work/detection-bridge-01'
SOURCES=('drone_nav/observation_channel.py','tools/channel_experiment.py','drone_nav/channel_lab.html',
         'drone_nav/detection_bridge.py','drone_nav/pinhole.py','drone_nav/occupancy.py','drone_nav/planning.py')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
canonical=lambda v:json.loads(json.dumps(v,allow_nan=False))


def load_input(raw):
    p=DetectionPacket(**raw['packet']);d=raw['context']
    c=None if d is None else SpatialContext(**dict(d,intrinsics=Intrinsics(**d['intrinsics']),pose=Pose(**d['pose'])))
    return p,c


def simulate(spec):
    ch=SurfaceEvidenceChannel(20,16,8,clock_id=spec['clock_id'])
    g=VoxelMap(20,16,8,ttl_ticks=100)
    # 受控几何夹具只隔离规划逻辑；不声称这些空闲格由本轮检测或真实照片产生。
    g.free_seen={tuple(cell):0 for cell in spec['controlled_free_voxels']}
    events=[]
    for event in spec['events']:
        receipt=None
        if event['input'] is not None:
            p,c=load_input(event['input']);receipt=ch.ingest(p,c,now_s=event['now_s'])
        decision=shadow_route(g,ch,layer=1,tick=0,now_s=event['now_s'],start=(11,7),goal=(17,7))
        if decision['flight_authorized']:raise ValueError('shadow channel cannot authorize flight')
        forbidden={tuple(c) for c in decision['base_blocked']+decision['restrictions']}
        if any(tuple(c) in forbidden for c in decision['route']):raise ValueError('route crosses restriction')
        if decision['route'] and not decision['baseline_route']:raise ValueError('overlay created unsupported free space')
        events.append(dict(title=event['title'],receipt=receipt,decision=decision))
    return canonical(events)


def changed(raw,name,t,done=None,empty=False):
    r=deepcopy(raw);r['packet'].update(frame_id=name,captured_at_s=t,completed_at_s=done if done is not None else t+.05)
    if empty:r['packet']['detections']=[]
    r['context'].update(frame_id=name,depth_at_s=t,pose_at_s=t)
    return r


def specs(parent):
    raw=read(SOURCE/'front/input.json');raw.pop('evaluation_truth')
    older=changed(raw,'older',.95)
    bad=changed(raw,'bad',1.25);bad['context']['frame_id']='wrong-source-frame'
    sequence=[dict(title='首次同步观测',now_s=1.2,input=raw),dict(title='同一帧重复到达',now_s=1.25,input=raw),
              dict(title='较早采集帧迟到',now_s=1.3,input=older),dict(title='错误帧不能更新缓存',now_s=1.35,input=bad),
              dict(title='新空检测不清除旧证据',now_s=1.45,input=changed(raw,'empty',1.3,empty=True)),
              dict(title='到期成为未知限制',now_s=1.51,input=None),
              dict(title='新独立帧恢复表面证据',now_s=1.8,input=changed(raw,'fresh',1.6,1.7)),
              dict(title='新证据再次到期',now_s=2.11,input=None)]
    result=[]
    for key,title,ys,events in [('wide','可绕行走廊',range(6,9),sequence),('narrow','单通道受阻',[7],[sequence[0],sequence[5]]),
                                ('unknown','基础地图全未知',[],[sequence[0]])]:
        result.append(dict(key=key,title=title,image_source='front/input.png',clock_id='synthetic-seconds',
                           controlled_free_voxels=[[x,y,1] for x in range(11,18) for y in ys],events=events))
    for c in parent['cases']:
        if c['kind']!='real-image':continue
        raw=read(SOURCE/c['input'])
        result.append(dict(key=c['key'],title='真实图缺少空间信息：保持未知',image_source=c['image'],
                           clock_id='offline-review-clock',controlled_free_voxels=[],
                           events=[dict(title='二维检测不能形成可通行地图',now_s=0,input=raw)]))
    return result


def run(out,verify=False):
    if not verify and out.exists():raise FileExistsError('refuse to overwrite '+str(out))
    check_manifest(SOURCE);parent=read(SOURCE/'report.json')
    for name,digest in parent['sources'].items():
        if sha(PROJECT/name)!=digest:raise ValueError('parent source differs')
    hashes={n:sha(PROJECT/n) for n in SOURCES}
    if verify:
        count=check_manifest(out);r=read(out/'report.json')
        if r['sources']!=hashes or r['parent_sha256']!=sha(SOURCE/'report.json') or r['protocol_sha256']!=sha(out/'protocol.md'):
            raise ValueError('source or protocol differs')
        for episode in r['episodes']:
            if simulate(read(out/episode['input']))!=episode['events']:raise ValueError('event replay differs')
        return dict(verified=True,files=count,episodes=len(r['episodes']),events=sum(len(e['events']) for e in r['episodes']))
    out.mkdir(parents=True,exist_ok=False)
    put(out/'protocol.md',(PROJECT/'docs/OBSERVATION_CHANNEL_PROTOCOL.md').read_bytes());episodes=[]
    for spec in specs(parent):
        events=simulate(spec);key=spec['key']
        put(out/key/'input.png',(SOURCE/spec['image_source']).read_bytes());dump(out/key/'input.json',spec)
        episodes.append(dict(key=key,title=spec['title'],input=key+'/input.json',image=key+'/input.png',events=events))
    for name in ('LICENSE','NOTICE','README.md','sources.json'):put(out/'model-source'/name,(SOURCE/'model-source'/name).read_bytes())
    if hashes!={n:sha(PROJECT/n) for n in SOURCES}:raise ValueError('source changed during experiment')
    r=dict(kind='shadow-observation-channel',sources=hashes,parent_sha256=sha(SOURCE/'report.json'),
           protocol_sha256=sha(out/'protocol.md'),width=20,height=16,layer=1,episodes=episodes)
    dump(out/'report.json',r)
    template=(PROJECT/'drone_nav/channel_lab.html').read_text(encoding='utf-8')
    put(out/'demo.html',template.replace('__DATA__',json.dumps(r,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return dict(episodes=len(episodes),events=sum(len(e['events']) for e in episodes))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
