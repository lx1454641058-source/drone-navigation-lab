"""来源：本项目原创。真实表面点/相机位姿统一到首个选中相机坐标系。"""
import argparse
from copy import deepcopy
from dataclasses import asdict,replace
import html
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.frame_transform import RigidFrameTransform,CameraPoseStamp,transform_delivery,finish_transform
from drone_nav.tum_rgbd import CLOCK,WORLD,load_observation,pose_from_record
from drone_nav.pinhole import Pose
from drone_nav.realvision import sha,dump
from tools.tum_rgbd_experiment import check_manifest,manifest,independent_projection
from tools.tum_bag_subset import verify_dataset

TARGET='tum_selected_first_camera_m'
CAMERA='tum-freiburg3-rgb'
OWN=('drone_nav/frame_transform.py','tests/test_frame_transform.py',
     'tools/frame_transform_experiment.py','docs/FRAME_TRANSFORM_PROTOCOL.md')


def sources(parent):
    for name,expected in parent['sources'].items():
        if sha(ROOT/name)!=expected: raise ValueError('frozen source differs: '+name)
    return dict(parent['sources'],**{name:sha(ROOT/name) for name in OWN})


def rotate(q,point,inverse=False):
    x,y,z,w=q; length=math.sqrt(sum(v*v for v in q)); x,y,z,w=(v/length for v in q)
    v=(-x,-y,-z) if inverse else (x,y,z)
    def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    first=cross(v,point); second=cross(v,first)
    return tuple(point[i]+2*w*first[i]+2*second[i] for i in range(3))


def independent(mapped,observation,anchor_record,current_record,transform):
    # Do not use the transform matrix to compute expected target coordinates.
    anchor=tuple(map(float,anchor_record[1:])); current=tuple(map(float,current_record[1:]))
    def local(world): return rotate(anchor[3:],tuple(a-b for a,b in zip(world,anchor[:3])),True)
    p=mapped['camera_pose']; target_pose=Pose(*(tuple(p[k]) for k in ('position','right','down','forward')))
    max_world=max_pixel=max_depth=max_inverse=0.; count=0
    expected_pose=[local(current[:3])]+[rotate(anchor[3:],rotate(current[3:],axis),True)
                    for axis in ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))]
    for key,expected in zip(('position','right','down','forward'),expected_pose):
        if max(abs(a-b) for a,b in zip(p[key],expected))>1e-10: raise ValueError('independent camera pose differs')
    for entry in mapped['observation']['projection']['observations']:
        for sample in entry['samples']:
            u,v=sample['pixel']; k=observation.frame.intrinsics
            depth=observation.frame.depth_z_m[v*k.width+u]
            if depth!=sample['depth_z_m']: raise ValueError('source depth differs')
            optical=((u-k.cx)*depth/k.fx,(v-k.cy)*depth/k.fy,depth)
            world=tuple(a+b for a,b in zip(current[:3],rotate(current[3:],optical)))
            if max(abs(a-b) for a,b in zip(world,sample['source_world_m']))>1e-10: raise ValueError('independent source point differs')
            max_world=max(max_world,max(abs(a-b) for a,b in zip(local(world),sample['world_m'])))
            pixel=target_pose.project(sample['world_m'],k)
            if pixel is None: raise ValueError('point moved behind camera')
            max_pixel=max(max_pixel,abs(pixel[0]-u),abs(pixel[1]-v)); max_depth=max(max_depth,abs(pixel[2]-depth))
            back=transform.inverse().point(sample['world_m'],source_frame=TARGET,target_frame=WORLD)
            max_inverse=max(max_inverse,max(abs(a-b) for a,b in zip(back,world))); count+=1
    if max_world>1e-10 or max_inverse>1e-10 or max_pixel>1e-8 or max_depth>1e-10:
        raise ValueError('point/pose/pixel invariant failed')
    return dict(samples=count,max_world_difference_m=max_world,max_pixel_difference=max_pixel,
                max_optical_depth_difference_m=max_depth,max_inverse_difference_m=max_inverse)


def rejected_cases(delivery,geometry,transform):
    def call(t=transform,g=geometry,now=None,target=TARGET):
        return transform_delivery(delivery,t,g,now_s=delivery['consumed_at_s'] if now is None else now,
                                  clock_id=CLOCK,target_frame=target)
    actions=[('inverse-direction',lambda:call(t=transform.inverse())),
             ('wrong-target-name',lambda:call(target='east_north_up_m')),
             ('millimetre-unit',lambda:replace(transform,unit='mm')),
             ('reflection-matrix',lambda:replace(transform,rotation=((-1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))),
             ('missing-reference-digest',lambda:replace(transform,reference_sha256='')),
             ('wrong-camera-position',lambda:call(g=replace(geometry,pose=replace(geometry.pose,
                  position=tuple(v+(.1 if i==0 else 0) for i,v in enumerate(geometry.pose.position)))))),
             ('expired-before-conversion',lambda:call(now=delivery['observation']['projection']['captured_at_s']+.51))]
    result=[]
    for name,action in actions:
        try: action()
        except ValueError as exc: result.append(dict(name=name,rejected=True,reason=str(exc)))
        else: result.append(dict(name=name,rejected=False))
    mapped=call(); late=finish_transform(mapped,now_s=mapped['valid_until_s']+.001,clock_id=CLOCK)
    result.append(dict(name='expired-after-conversion',rejected=not late['available'] and late['result'] is None,reason=late['reason']))
    return result


def page(report):
    colours=['#126da8','#b74723','#7d469c','#258263','#9a7214','#3454a4']
    def plot(horizontal,vertical,title):
        points=[]
        for index,row in enumerate(report['frames']):
            mapped=row['mapped']
            points.extend((index,s['world_m'],False) for o in mapped['observation']['projection']['observations'] for s in o['samples'])
            points.append((index,mapped['camera_pose']['position'],True))
        xs=[p[1][horizontal] for p in points]; ys=[p[1][vertical] for p in points]
        lo_x,hi_x=min(xs)-.5,max(xs)+.5; lo_y,hi_y=min(ys)-.5,max(ys)+.5
        def xy(p): return 55+(p[horizontal]-lo_x)/(hi_x-lo_x)*490,300-(p[vertical]-lo_y)/(hi_y-lo_y)*260
        marks=[]
        for index,p,camera in points:
            x,y=xy(p)
            marks.append(f'<circle class="frame-point" data-frame="{index}" cx="{x:.3f}" cy="{y:.3f}" r="{6 if camera else 2.5}" fill="{colours[index]}" stroke="{"#111" if camera else "none"}" style="display:{"inline" if index==0 else "none"}"><title>{report["frames"][index]["offset_s"]} 秒 · {"相机" if camera else "表面点"} · {html.escape(str(p))}</title></circle>')
        return f'<article><h2>{title}</h2><svg viewBox="0 0 580 350" role="img" aria-label="{title}"><rect x="55" y="40" width="490" height="260" fill="#f2f5f8"/><path d="M55 40 V300 H545" stroke="#6c7a87" fill="none"/>{"".join(marks)}<text x="55" y="325">{lo_x:.2f} m</text><text x="465" y="325">{hi_x:.2f} m</text><text x="5" y="45">{hi_y:.2f}</text><text x="5" y="298">{lo_y:.2f}</text></svg></article>'
    options=''.join(f'<option value="{i}">{r["offset_s"]} 秒：{r["audit"]["samples"]} 个表面点</option>' for i,r in enumerate(report['frames']))
    rows=''.join(f'<tr><td>{r["offset_s"]} s</td><td>{r["audit"]["samples"]}</td><td>{r["elapsed_s"]*1000:.3f} ms</td><td>{r["finished"]["age_s"]*1000:.2f} ms</td><td>{"可查看" if r["finished"]["available"] else "过期"}</td><td>{r["audit"]["max_pixel_difference"]:.2e}</td></tr>' for r in report['frames'])
    failures=''.join(f'<li>{r["name"]}：{"拒绝" if r["rejected"] else "未拒绝"} — {html.escape(r.get("reason",""))}</li>' for r in report['negative_cases'])
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实观测的相机参考坐标</title><style>body{{font:16px/1.7 system-ui;background:#edf3f7;color:#173148;margin:0;padding:24px}}main{{max-width:1220px;margin:auto}}section,article{{background:white;border-radius:12px;padding:22px;margin:18px 0}}.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}}svg{{width:100%}}h2{{font-size:20px}}select{{font:inherit;padding:9px;max-width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #dbe3e9;text-align:left;white-space:nowrap}}.scroll{{overflow:auto}}.warn{{color:#874414}}a{{color:#1263a1}}pre{{overflow:auto}}</style>
<main><h1>把观测统一到首个选中相机的坐标系</h1><section><p>原点：所选 2 秒图像对应的动捕相机位置。X 向相机右、Y 向下、Z 向前，单位米。</p><p class="warn">这是一套可复算的研究坐标，不是重力对齐或无人机导航标定。旋转和平移不提高检测精度，不授权地图更新或飞行。</p><label for="frame">显示时间：</label><select id="frame">{options}<option value="all">历史叠加（不同时间，不代表当前地图）</option></select><p id="view-note">当前显示所选第一帧；大圆为相机，小圆为检测框内表面点，可能含背景。</p></section>
<div class="plots">{plot(0,2,'水平轴：右 X；竖轴：前 Z')}{plot(0,1,'水平轴：右 X；竖轴：下 Y（非重力）')}</div>
<section><h2>逐帧变换与独立反投影</h2><p>使用保存的消费结果，加上本轮实测转换时间进行离线时效重放；不是新的在线全流程测速。原几何读取及核验耗时不计入转换阶段。</p><div class="scroll"><table><tr><th>图像</th><th>采样点</th><th>转换阶段</th><th>完成年龄</th><th>研究证据</th><th>最大像素差</th></tr>{rows}</table></div></section>
<section><h2>八种错误与过期检查</h2><ul>{failures}</ul></section><section><h2>参考变换</h2><pre>{html.escape(json.dumps(report['transform'],ensure_ascii=False,indent=2))}</pre><p>摘要用于定位原选择/位姿记录，不能证明物理标定精度。</p><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定方案</a> · <a href="dataset-sources.json">TUM 数据署名与许可</a></p></section></main>
<script>document.getElementById('frame').addEventListener('change',function(){{document.querySelectorAll('.frame-point').forEach(p=>p.style.display=this.value==='all'||p.dataset.frame===this.value?'inline':'none');document.getElementById('view-note').textContent=this.value==='all'?'历史叠加包含不同时间的观测，不能视为当前地图或动态目标轨迹。':'仅显示所选时刻。大圆为相机，小圆为框内可见表面；无点不代表空闲。';}});</script></html>'''


def run(dataset,parent_dir,output,verify=False):
    check_manifest(dataset); check_manifest(parent_dir); verify_dataset(dataset)
    parent=json.loads((parent_dir/'report.json').read_text(encoding='utf-8'))
    selection=json.loads((dataset/'selection.json').read_text(encoding='utf-8'))
    anchor=selection['frames'][0]
    transform=RigidFrameTransform.from_reference_pose(pose_from_record(anchor['pose']),source_frame=WORLD,target_frame=TARGET,
        reference_id=anchor['rgb'][1],reference_sha256=sha(dataset/'selection.json'))
    fixed=dict(sources=sources(parent),parent_sha256=sha(parent_dir/'report.json'),dataset_sha256=sha(dataset/'manifest.json'),
               transform=asdict(transform))
    # Normalize tuple fields exactly as they appear after JSON serialization.
    fixed=json.loads(json.dumps(fixed))
    saved=None
    if verify:
        count=check_manifest(output); saved=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if any(saved[k]!=v for k,v in fixed.items()): raise ValueError('source/reference identity differs')
        if len(saved['frames'])!=6: raise ValueError('frame count differs')
    else:
        output.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(ROOT/'docs/FRAME_TRANSFORM_PROTOCOL.md',output/'protocol.md')
        shutil.copyfile(dataset/'sources.json',output/'dataset-sources.json')
    report=dict(**fixed,frames=[])
    for index,(selected,original) in enumerate(zip(selection['frames'],parent['live'])):
        observation=load_observation(dataset,selected,selection['origin_s'])
        delivery=original['consumption']['answer']; untouched=deepcopy(delivery)
        geometry=CameraPoseStamp(observation.frame_id,CAMERA,CLOCK,observation.pose_at_s,observation.frame.pose,observation.frame.intrinsics)
        now=delivery['consumed_at_s']
        start=perf_counter(); mapped=transform_delivery(delivery,transform,geometry,now_s=now,clock_id=CLOCK,target_frame=TARGET)
        elapsed=perf_counter()-start
        if saved: elapsed=saved['frames'][index]['elapsed_s']
        if not math.isfinite(elapsed) or elapsed<0: raise ValueError('invalid stage timing')
        finished=finish_transform(mapped,now_s=now+elapsed,clock_id=CLOCK)
        if delivery!=untouched: raise ValueError('conversion changed original delivery')
        audit=independent(mapped,observation,anchor['pose'],selected['pose'],transform)
        if independent_projection(observation,delivery['observation'],selected['pose'])!=audit['samples']:
            raise ValueError('source/target sample count differs')
        row=dict(index=index,offset_s=selected['offset_s'],elapsed_s=elapsed,mapped=mapped,finished=finished,audit=audit)
        # Saved JSON uses arrays; compare that representation without loose floats.
        row=json.loads(json.dumps(row))
        if saved and row!=saved['frames'][index]: raise ValueError('coordinate/time replay differs')
        report['frames'].append(row)
        if index==0: report['negative_cases']=rejected_cases(delivery,geometry,transform)
        print('verified' if verify else 'converted',selected['offset_s'],audit['samples'],finished['available'],flush=True)
    if not all(r['rejected'] for r in report['negative_cases']): raise ValueError('negative case unexpectedly passed')
    if verify:
        if report!=saved or page(report)!=(output/'demo.html').read_text(encoding='utf-8') or check_manifest(output)!=count:
            raise ValueError('negative cases/report/page differ or archive changed')
        return dict(verified=True,frames=6,samples=sum(r['audit']['samples'] for r in report['frames']),negative_cases=8,files=count,sources=len(fixed['sources']))
    if sources(parent)!=fixed['sources']: raise ValueError('source changed while running')
    dump(output/'report.json',report); (output/'demo.html').write_text(page(report),encoding='utf-8')
    dump(output/'manifest.json',manifest(output))
    return dict(frames=6,samples=sum(r['audit']['samples'] for r in report['frames']),negative_cases=8,
                available=sum(r['finished']['available'] for r in report['frames']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=ROOT/'work/tum-rgbd-bag-input-01')
    parser.add_argument('--parent',type=Path,default=ROOT/'work/timed-inbox-01')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.parent,args.output,args.verify),indent=2),flush=True)
