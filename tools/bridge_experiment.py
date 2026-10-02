"""来源：本项目原创。同步检测投影与真实航拍仅二维证据的开发归档。"""
import argparse
from dataclasses import asdict,replace
import json
from math import sqrt
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.detection_bridge import DetectionPacket,SpatialContext,BridgeConfig,project_detections
from drone_nav.pinhole import Intrinsics,Pose
from drone_nav.raycast import render,demo_world
from drone_nav.vision_experiment import train_model
from drone_nav.imaging import png_bytes
from drone_nav.realvision import dump,sha
from tools.setup_vision import put
from tools.object_probe import check_manifest

SOURCE=PROJECT/'work/query-dedup-holdout-01'
SOURCES=('tools/bridge_experiment.py','drone_nav/detection_bridge.py','drone_nav/bridge_lab.html',
         'drone_nav/pinhole.py','drone_nav/raycast.py','drone_nav/classifier.py','drone_nav/vision_experiment.py',
         'drone_nav/imaging.py','drone_nav/rendering.py')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def components(labels,width,height):
    """四邻接人物颜色区域转框，不读取场景对象或真值标签。score=1 仅为固定接口占位。"""
    pending={i for i,label in enumerate(labels) if label==4};boxes=[]
    while pending:
        seed=min(pending);pending.remove(seed);stack=[seed];pixels=[]
        while stack:
            i=stack.pop();pixels.append(i);x,y=i%width,i//width
            for xx,yy in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                j=yy*width+xx
                if 0<=xx<width and 0<=yy<height and j in pending:pending.remove(j);stack.append(j)
        xs=[i%width for i in pixels];ys=[i//width for i in pixels]
        boxes.append(dict(id=len(boxes),group='person',score=1.0,box=[min(xs),min(ys),max(xs)+1,max(ys)+1]))
    return tuple(boxes)


def serialize(packet,context,now=1.2,clock='synthetic-seconds'):
    return dict(packet=asdict(packet),context=None if context is None else asdict(context),now_s=now,now_clock_id=clock)


def replay(raw):
    p=DetectionPacket(**raw['packet']);data=raw['context']
    c=None if data is None else SpatialContext(**dict(data,intrinsics=Intrinsics(**data['intrinsics']),pose=Pose(**data['pose'])))
    return project_detections(p,c,now_s=raw['now_s'],now_clock_id=raw['now_clock_id'])


def metrics(result,truth,width):
    samples=[s for o in result['observations'] for s in o['samples']];errors=[];matched=0
    if truth is not None:
        for o in result['observations']:
            for s in o['samples']:
                u,v=s['pixel'];i=v*width+u;t=truth['points'][i]
                if t is not None:errors.append(sqrt(sum((a-b)**2 for a,b in zip(s['world_m'],t))))
                if truth['labels'][i]==4 and o['detection']['group']=='person':matched+=1
    return dict(detections=len(result['observations']),surface_samples=len(samples),
                geometry_max_error_m=max(errors) if errors else None,
                synthetic_person_surface_samples=matched if truth is not None else None)


def run(out,verify=False):
    if not verify and out.exists():raise FileExistsError('refuse to overwrite '+str(out))
    check_manifest(SOURCE);parent=read(SOURCE/'report.json')
    for name,digest in parent['sources'].items():
        if sha(PROJECT/name)!=digest:raise ValueError('parent source changed')
    hashes={n:sha(PROJECT/n) for n in SOURCES}
    if verify:
        count=check_manifest(out);report=read(out/'report.json')
        if report['sources']!=hashes or report['parent_sha256']!=sha(SOURCE/'report.json') or report['protocol_sha256']!=sha(out/'protocol.md'):
            raise ValueError('source or protocol mismatch')
        for case in report['cases']:
            raw=read(out/case['input']);result=replay(raw)
            if result!=case['result'] or metrics(result,raw.get('evaluation_truth'),raw['packet']['width'])!=case['metrics']:
                raise ValueError('input replay differs: '+case['key'])
        return dict(verified=True,files=count,cases=len(report['cases']))
    out.mkdir(parents=True,exist_ok=False);cases=[]
    put(out/'protocol.md',(PROJECT/'docs/DETECTION_BRIDGE_PROTOCOL.md').read_bytes())
    model=train_model();dump(out/'synthetic-model.json',model.to_dict())

    def save(key,title,raw,image,expected,kind):
        result=replay(raw)
        if result['status']!=expected:raise ValueError('unexpected case status '+key+': '+result['status'])
        if result['flight_authorized'] or result['free_space_evidence']:raise ValueError('adapter cannot authorize free space')
        for o in result['observations']:
            if result['status']=='CONTEXT_REJECTED' and o['samples']:raise ValueError('rejected context projected points')
        put(out/key/'input.png',image);dump(out/key/'input.json',raw)
        cases.append(dict(key=key,title=title,kind=kind,width=raw['packet']['width'],height=raw['packet']['height'],input=key+'/input.json',image=key+'/input.png',
                          result=result,metrics=metrics(result,raw.get('evaluation_truth'),raw['packet']['width'])))

    poses=[('front','正向观察',(14.3,4,5),(14.3,7.8,.8)),('side','侧向观察',(18,8,4),(14.3,7.8,.8)),('overhead','俯视观察',(14.3,7.8,7),(14.3,7.8,0))]
    base=None
    for key,title,pos,target in poses:
        k=Intrinsics();frame,truth=render(demo_world(),Pose.look_at(pos,target),k)
        detections=components(model.predict_image(frame.rgb),k.width,k.height)
        if not detections:raise ValueError('synthetic model did not detect a person')
        packet=DetectionPacket(key,'synthetic-camera',k.width,k.height,detections,'synthetic_color_components; score=1 is not probability',1,1.1,'synthetic-seconds')
        context=SpatialContext(key,'synthetic-camera','synthetic-seconds',1,1,k,frame.pose,frame.depth_z_m,True,
                               'camera_optical_axis_z_m','raycast_simulated_depth','exact_simulated_capture_pose','synthetic-pinhole-v1')
        raw=serialize(packet,context);raw['evaluation_truth']=asdict(truth)
        image=png_bytes(k.width,k.height,frame.rgb)
        save(key,title,raw,image,'SAMPLES_AVAILABLE','synthetic')
        if key=='front':base=(packet,context,raw['evaluation_truth'],image)
    p,c,truth,image=base
    variants=[('stale','推理完成时拍摄已过期',replace(p,completed_at_s=1.8),c,1.81,'CONTEXT_REJECTED'),
              ('depth-desync','深度与图像不同步',p,replace(c,depth_at_s=.8),1.2,'CONTEXT_REJECTED'),
              ('pose-desync','错误时刻的位姿',p,replace(c,pose_at_s=.8),1.2,'CONTEXT_REJECTED'),
              ('unregistered','深度未配准',p,replace(c,registered_to_rgb=False),1.2,'CONTEXT_REJECTED'),
              ('range-convention','错误深度约定',p,replace(c,depth_convention='ray_range_m'),1.2,'CONTEXT_REJECTED'),
              ('wrong-frame','串入另一帧深度',p,replace(c,frame_id='other'),1.2,'CONTEXT_REJECTED'),
              ('wrong-clock','深度时钟不一致',p,replace(c,clock_id='other-clock'),1.2,'CONTEXT_REJECTED'),
              ('missing-depth','全部深度缺失',p,replace(c,depth_z_m=(None,)*len(c.depth_z_m)),1.2,'NO_SPATIAL_SAMPLES'),
              ('empty','空检测不证明无障碍',replace(p,detections=()),c,1.2,'NO_SPATIAL_SAMPLES'),
              ('future','处理完成时间在未来',replace(p,completed_at_s=2),c,1.2,'CONTEXT_REJECTED')]
    for key,title,packet,context,now,expected in variants:
        raw=serialize(packet,context,now);raw['evaluation_truth']=truth
        save(key,title,raw,image,expected,'synthetic-fault')
    for case in parent['analysis']['cases']:
        original=next(r for r in parent['analysis']['results'] if r['key']==case['key'] and r['arm']=='tinyformer-query')
        detections=tuple(dict(id=i,group=b['group'],score=b['score'],box=[b[k] for k in ('x1','y1','x2','y2')]) for i,b in enumerate(original['boxes']))
        packet=DetectionPacket(case['key'],'unknown-camera',case['width'],case['height'],detections,'archived_tinyformer_query')
        raw=serialize(packet,None,0,'offline-review-clock')
        raw['parent_frame']=case['key'];raw['parent_sha256']=sha(SOURCE/'report.json')
        save(case['key'],'真实航拍：缺少空间与时间标定',raw,(SOURCE/case['key']/'input.png').read_bytes(),'CONTEXT_REJECTED','real-image')
    for name in ('LICENSE','NOTICE','README.md','sources.json'):put(out/'model-source'/name,(SOURCE/'model-source'/name).read_bytes())
    if hashes!={n:sha(PROJECT/n) for n in SOURCES}:raise ValueError('source changed during experiment')
    report=dict(kind='detection-spatial-bridge',config=asdict(BridgeConfig()),sources=hashes,parent_sha256=sha(SOURCE/'report.json'),
                protocol_sha256=sha(out/'protocol.md'),cases=cases)
    dump(out/'report.json',report)
    template=(PROJECT/'drone_nav/bridge_lab.html').read_text(encoding='utf-8')
    put(out/'demo.html',template.replace('__DATA__',json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return dict(cases=len(cases),synthetic=sum(c['kind']!='real-image' for c in cases),real_images=sum(c['kind']=='real-image' for c in cases),
                surface_samples=sum(c['metrics']['surface_samples'] for c in cases),max_geometry_error_m=max((c['metrics']['geometry_max_error_m'] or 0) for c in cases))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
