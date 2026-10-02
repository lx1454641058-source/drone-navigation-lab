"""来源：本项目原创。真实 RGB-D 的固定体积检查与独立四元数核验。"""
import argparse
from collections import Counter
from dataclasses import asdict
import html
from itertools import product
import json
from math import ceil,floor,sqrt
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.depth_volume import DepthVolumeInspector,QueryVolume
from drone_nav.packed_rgbd import load_packed
from drone_nav.tum_rgbd import CLOCK,INTRINSICS
from drone_nav.realvision import sha,dump
from tools.frame_transform_experiment import rotate
from tools.framed_surface_map_experiment import reference
from tools.tum_rgbd_experiment import manifest,check_manifest
from tools.tum_bag_subset import verify_dataset

OWN=('drone_nav/depth_volume.py','tests/test_depth_volume.py',
     'tools/depth_volume_experiment.py','docs/DEPTH_VOLUME_PROTOCOL.md')
BOXES=(('中间近处',(-.25,-.25,1.),(.25,.25,2.)),('中间远处',(-.5,-.5,2.),(.5,.5,5.)),
       ('左侧远处',(-1.25,-.5,2.),(-.25,.5,5.)),('右侧远处',(.25,-.5,2.),(1.25,.5,5.)))
LABELS={'MEASURED_SURFACE':'体积内测得表面','INSUFFICIENT_DEPTH':'缺测或量程不足',
        'FOREGROUND_OR_OCCLUSION':'前景、遮挡或余量不足','SAMPLED_BEYOND_VOLUME':'采样位于远处，覆盖未知',
        'INSUFFICIENT_VIEW':'视野不足','CONTEXT_REJECTED':'观测已过期'}


def sources(parent):
    for name,expected in parent['sources'].items():
        if sha(ROOT/name)!=expected: raise ValueError('frozen source differs: '+name)
    return dict(parent['sources'],**{name:sha(ROOT/name) for name in OWN})


def independent(result,volume,observation,anchor_record,current_record,semantic):
    anchor=tuple(map(float,anchor_record[1:])); current=tuple(map(float,current_record[1:]))
    k=observation.frame.intrinsics
    def to_camera(local):
        world=tuple(a+b for a,b in zip(anchor[:3],rotate(anchor[3:],local)))
        return rotate(current[3:],tuple(a-b for a,b in zip(world,current[:3])),True)
    corners=[to_camera(point) for point in product(*zip(volume.lower,volume.upper))]
    if any(p[2]<=0 for p in corners):
        if result['reasons']!=['VOLUME_NOT_FULLY_IN_FRONT']: raise ValueError('independent front clipping differs')
        return dict(checked_pixels=0,max_witness_difference_m=0.)
    projected=[(k.fx*p[0]/p[2]+k.cx,k.fy*p[1]/p[2]+k.cy,p[2]) for p in corners]
    window=[ceil(min(p[0] for p in projected)-.5-1e-9),ceil(min(p[1] for p in projected)-.5-1e-9),
            floor(max(p[0] for p in projected)+.5+1e-9),floor(max(p[1] for p in projected)+.5+1e-9)]
    far=max(p[2] for p in projected)
    if result['window']!=window or abs(result['far_z_m']-far)>1e-10: raise ValueError('independent projection differs')
    x0,y0,x1,y1=window
    if x0<0 or y0<0 or x1>=k.width or y1>=k.height:
        if result['reasons']!=['INCOMPLETE_VOLUME_VIEW']: raise ValueError('independent image clipping differs')
        return dict(checked_pixels=0,max_witness_difference_m=0.)
    origin=rotate(anchor[3:],tuple(a-b for a,b in zip(current[:3],anchor[:3])),True)
    axes=[rotate(anchor[3:],rotate(current[3:],axis),True) for axis in ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))]
    def local_point(u,v,z):
        optical=((u-k.cx)*z/k.fx,(v-k.cy)*z/k.fy,z)
        return tuple(origin[i]+sum(axes[j][i]*optical[j] for j in range(3)) for i in range(3))
    def inside(p): return all(volume.lower[i]-1e-9<=p[i]<=volume.upper[i]+1e-9 for i in range(3))
    counts=dict(checked_pixels=0,missing_pixels=0,out_of_range_pixels=0,foreground_pixels=0,inside_surface_pixels=0)
    witnesses=[]
    for v in range(y0,y1+1):
        for u in range(x0,x1+1):
            counts['checked_pixels']+=1; z=observation.frame.depth_z_m[v*k.width+u]
            if z is None: counts['missing_pixels']+=1; continue
            length=z*sqrt(1+((u-k.cx)/k.fx)**2+((v-k.cy)/k.fy)**2)
            if length>30.: counts['out_of_range_pixels']+=1; continue
            if z-.05<=far+1e-9: counts['foreground_pixels']+=1
            p=local_point(u,v,z)
            if inside(p):
                counts['inside_surface_pixels']+=1
                if len(witnesses)<8: witnesses.append(([u,v],z,p))
    counts['semantic_surface_samples']=sum(inside(local_point(*s['pixel'],s['depth_z_m']))
        for entry in semantic['observations'] for s in entry['samples'])
    if any(result[key]!=value for key,value in counts.items()): raise ValueError('independent depth/semantic counts differ')
    if len(witnesses)!=len(result['witnesses']): raise ValueError('independent witness count differs')
    error=0.
    for (pixel,z,p),actual in zip(witnesses,result['witnesses']):
        if pixel!=actual['pixel'] or z!=actual['depth_z_m']: raise ValueError('independent witness identity differs')
        error=max(error,max(abs(a-b) for a,b in zip(p,actual['world_m'])))
    if error>1e-10: raise ValueError('independent witness coordinates differ')
    return dict(**counts,max_witness_difference_m=error)


def page(report):
    options=''.join(f'<option value="{i}">{r["offset_s"]} 秒图像</option>' for i,r in enumerate(report['frames']))
    frames=[]
    for i,row in enumerate(report['frames']):
        cards=[]
        for q in row['queries']:
            r=q['result']; box=''
            if 'window' in r:
                x0,y0,x1,y1=r['window']
                box=f'<rect x="{x0-.5}" y="{y0-.5}" width="{x1-x0+1}" height="{y1-y0+1}" fill="none" stroke="#39ccff" stroke-width="3"/>'
            marks=''.join(f'<circle cx="{w["pixel"][0]}" cy="{w["pixel"][1]}" r="3" fill="#ffad32"/>' for w in r['witnesses'])
            cards.append(f'<article><h3>{html.escape(r["volume"])}</h3><p class="status">{LABELS[r["status"]]}</p><svg viewBox="0 0 640 480" role="img" aria-label="{html.escape(r["volume"])}查询投影与表面见证点"><image href="{row["image"]}" width="640" height="480"/>{box}{marks}</svg><p>读取 {r["checked_pixels"]:,} 像素 · 缺测 {r["missing_pixels"]:,}<br>体积内表面 {r["inside_surface_pixels"]:,} 像素 · 语义采样 {r["semantic_surface_samples"]}</p><details><summary>范围与检查依据</summary><pre>{html.escape(json.dumps(dict(volume=q["volume"],result=r,audit=q["audit"]),ensure_ascii=False,indent=2))}</pre></details></article>')
        frames.append(f'<div class="frame cards" data-frame="{i}" style="display:{"grid" if i==0 else "none"}">{"".join(cards)}</div>')
    stats=''.join(f'<li>{LABELS[key]}：{count} 项</li>' for key,count in sorted(report['summary']['statuses'].items()))
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实深度的候选空间检查</title><style>body{margin:0;padding:24px;background:#eef2f6;color:#193347;font:16px/1.7 system-ui}main{max-width:1200px;margin:auto}section,article{background:white;border-radius:12px;padding:22px;margin:18px 0}.cards{grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:20px}article{margin:0}h1{font-size:30px}h3{margin:0}svg{width:100%;border-radius:5px}select{font:inherit;padding:9px}.status{color:#914919;font-weight:600}pre{max-height:340px;overflow:auto;font-size:12px}a{color:#175b96}.note{color:#70431d}</style>
<main><h1>真实深度的候选空间检查</h1><p>六帧原图 · 四个固定三维框 · 24 项离线查询</p><section><h2>把“有检测框”扩展到检查整个投影区域</h2><p>蓝线是候选三维框的投影矩形，橙点是体积内最多八个实测表面见证点。检测模型没有识别到物体时，几何检查仍然读取深度。</p><p class="note">候选框固定在所选首帧相机的右/下/前坐标，后续相机移动会改变投影。框不代表真实无人机完整机体轨迹。无点、背景深度较远或空检测都不证明可飞。</p><label for="frame">图像时刻：</label><select id="frame">'''+options+'''</select></section>'''+''.join(frames)+'''<section><h2>本次结果</h2><ul>'''+stats+'''</ul><p>缺失深度、视野不足、前景遮挡和体积内实测表面分别记录；类别有优先级，展开依据可看到同时存在的条件。体积内点是测量证据，不是碰撞真值或确认人物。</p><p>普通深度是点采样，仍缺像素间覆盖、实际误差标定、动态变化和导航机体条件；全部查询均不授权飞行。</p></section><section><h2>复核资料</h2><p>采用保存的转换完成时间，未进行新模型推理或本阶段实时测速。独立原始四元数公式重算投影、逐像素计数及见证坐标。</p><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定方案</a> · <a href="dataset-sources.json">TUM 来源与许可</a></p></section></main><script>document.getElementById('frame').addEventListener('change',function(){document.querySelectorAll('.frame').forEach(e=>e.style.display=e.dataset.frame===this.value?'grid':'none');});</script></html>'''


def run(dataset,converted_dir,parent_dir,output,verify=False):
    check_manifest(dataset); check_manifest(converted_dir); check_manifest(parent_dir); verify_dataset(dataset)
    converted=json.loads((converted_dir/'report.json').read_text(encoding='utf-8'))
    parent=json.loads((parent_dir/'report.json').read_text(encoding='utf-8'))
    selection=json.loads((dataset/'selection.json').read_text(encoding='utf-8'))
    if parent['parent_sha256']!=sha(converted_dir/'report.json') or converted['dataset_sha256']!=sha(dataset/'manifest.json'):
        raise ValueError('archive ancestry differs')
    fixed=sources(parent); transform=reference(converted)
    if [r['offset_s'] for r in selection['frames']]!=[2,4,6,8,10,12] or len(converted['frames'])!=6:
        raise ValueError('fixed input selection differs')
    report=dict(sources=fixed,parent_sha256=sha(parent_dir/'report.json'),converted_sha256=sha(converted_dir/'report.json'),
        dataset_sha256=sha(dataset/'manifest.json'),timing_scope='offline_saved_conversion_completion_times',
        new_model_calls=0,physical_flights=0,frames=[])
    for i,(selected,saved) in enumerate(zip(selection['frames'],converted['frames'])):
        observation=load_packed(dataset,selected,selection['origin_s'])
        inspector=DepthVolumeInspector(observation,saved['finished'],transform,camera_id='tum-freiburg3-rgb',
            clock_id=CLOCK,expected_intrinsics=INTRINSICS)
        now=saved['finished']['now_s']; queries=[]
        for name,lo,hi in BOXES:
            volume=QueryVolume(name,transform.target_frame,lo,hi)
            result=inspector.inspect(volume,now_s=now,clock_id=CLOCK)
            audit=independent(result,volume,observation,selection['frames'][0]['pose'],selected['pose'],saved['mapped']['observation']['projection'])
            queries.append(dict(volume=asdict(volume),result=result,audit=audit))
        row=dict(offset_s=selected['offset_s'],image=f'frame-{i}.png',image_sha256=sha(dataset/selected['rgb'][1]),queries=queries)
        report['frames'].append(row)
        print('checked',selected['offset_s'],[(q['result']['status'],q['result']['inside_surface_pixels']) for q in queries],flush=True)
    all_queries=[q for r in report['frames'] for q in r['queries']]
    report['summary']=dict(queries=len(all_queries),statuses=dict(Counter(q['result']['status'] for q in all_queries)),
        checked_pixels=sum(q['result']['checked_pixels'] for q in all_queries),
        missing_pixels=sum(q['result']['missing_pixels'] for q in all_queries),
        inside_surface_pixels=sum(q['result']['inside_surface_pixels'] for q in all_queries),
        semantic_surface_samples=sum(q['result']['semantic_surface_samples'] for q in all_queries),
        max_witness_difference_m=max(q['audit']['max_witness_difference_m'] for q in all_queries))
    if sources(parent)!=fixed: raise ValueError('source changed')
    report=json.loads(json.dumps(report))
    if verify:
        count=check_manifest(output)
        if report!=json.loads((output/'report.json').read_text(encoding='utf-8')): raise ValueError('depth query replay differs')
        if page(report)!=(output/'demo.html').read_text(encoding='utf-8'): raise ValueError('page replay differs')
        for row in report['frames']:
            if sha(output/row['image'])!=row['image_sha256']: raise ValueError('original image differs')
        if (output/'protocol.md').read_bytes()!=(ROOT/'docs/DEPTH_VOLUME_PROTOCOL.md').read_bytes(): raise ValueError('protocol differs')
        if (output/'dataset-sources.json').read_bytes()!=(dataset/'sources.json').read_bytes(): raise ValueError('attribution differs')
        if check_manifest(output)!=count: raise ValueError('archive changed')
    else:
        output.mkdir(parents=True,exist_ok=False)
        for row,selected in zip(report['frames'],selection['frames']): shutil.copyfile(dataset/selected['rgb'][1],output/row['image'])
        shutil.copyfile(ROOT/'docs/DEPTH_VOLUME_PROTOCOL.md',output/'protocol.md')
        shutil.copyfile(dataset/'sources.json',output/'dataset-sources.json')
        dump(output/'report.json',report); (output/'demo.html').write_text(page(report),encoding='utf-8')
        dump(output/'manifest.json',manifest(output)); count=check_manifest(output)
    return dict(verified=verify,files=count,sources=len(fixed),**report['summary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=ROOT/'work/tum-rgbd-bag-input-01')
    parser.add_argument('--converted',type=Path,default=ROOT/'work/frame-transform-01')
    parser.add_argument('--parent',type=Path,default=ROOT/'work/framed-surface-map-01')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.converted,args.parent,args.output,args.verify),ensure_ascii=False,indent=2),flush=True)
