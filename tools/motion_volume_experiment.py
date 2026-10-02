"""来源：本项目原创。真实深度下的候选运动停止范围和 A* 对照。"""
import argparse
from collections import Counter
from dataclasses import replace
import html
import json
from math import sqrt
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.depth_volume import DepthVolumeInspector,QueryVolume
from drone_nav.motion import MotionConfig
from drone_nav.motion_volume import RestMotionCandidate,evaluate_motion,planning_comparison
from drone_nav.packed_rgbd import load_packed
from drone_nav.tum_rgbd import CLOCK,INTRINSICS
from drone_nav.realvision import sha,dump
from tools.depth_volume_experiment import independent,LABELS
from tools.framed_surface_map_experiment import reference
from tools.tum_rgbd_experiment import check_manifest,manifest
from tools.tum_bag_subset import verify_dataset

OWN=('drone_nav/motion_volume.py','tests/test_motion_volume.py','tools/motion_volume_experiment.py',
     'docs/MOTION_VOLUME_PROTOCOL.md','drone_nav/motion.py','drone_nav/planning.py')
ORIGIN=(-.5,0.,1.25)


def sources(parent):
    for name,expected in parent['sources'].items():
        if sha(ROOT/name)!=expected: raise ValueError('frozen source differs: '+name)
    return dict(parent['sources'],**{name:sha(ROOT/name) for name in OWN})


def independent_stop(result):
    e=result['envelope']; c=e['config']; start=e['candidate']['start']; end=e['candidate']['end']
    distance=sqrt(sum((b-a)**2 for a,b in zip(start,end))); direction=[(b-a)/distance for a,b in zip(start,end)]
    a=c['acceleration_mps2']; b=c['brake_mps2']; tau=c['reaction_s']
    peak=min(c['max_speed_mps'],sqrt(2*distance*a*b/(a+b)))
    ta=peak/a; tb=peak/b; tc=max(0.,(distance-peak*peak/(2*a)-peak*peak/(2*b))/peak)
    total=ta+tc+tb; length=distance+peak*tau+c['distance_margin_m']
    padding=c['body_radius_m']+2*e['error_bound_m']
    far=[x+v*length for x,v in zip(start,direction)]
    lower=[min(x,y)-padding for x,y in zip(start,far)]; upper=[max(x,y)+padding for x,y in zip(start,far)]
    comparisons=[(e['required_distance_m'],length),(e['normal_duration_s'],total),
        (e['latest_stop_s'],e['departure_at_s']+total+tau),(e['relative_padding_m'],padding)]
    comparisons.extend(zip(e['volume']['lower'],lower)); comparisons.extend(zip(e['volume']['upper'],upper))
    if any(abs(x-y)>1e-10 for x,y in comparisons): raise ValueError('independent stopping envelope differs')
    expires=e['departure_at_s']+total+tau>result['geometry']['valid_until_s']+1e-9
    if result['expires_before_stop']!=expires: raise ValueError('independent stopping deadline differs')
    max_distance=max_stop_time=0.
    for i in range(51):
        t=total*i/50
        if t<=ta: x=.5*a*t*t; v=a*t
        elif t<=ta+tc: x=peak*peak/(2*a)+peak*(t-ta); v=peak
        else:
            elapsed=t-ta-tc; v=max(0.,peak-b*elapsed)
            x=peak*peak/(2*a)+peak*tc+peak*elapsed-.5*b*elapsed*elapsed
        stop=x+v*tau+v*v/(2*b); stop_time=t+tau+v/b
        if stop>length+1e-9 or stop_time>total+tau+1e-9: raise ValueError('fault trajectory exceeds envelope')
        for j in range(3):
            position=start[j]+direction[j]*stop
            if position-padding<lower[j]-1e-9 or position+padding>upper[j]+1e-9:
                raise ValueError('stopped body exceeds volume')
        max_distance=max(max_distance,stop); max_stop_time=max(max_stop_time,stop_time)
    return dict(fault_times_checked=51,max_fault_stop_distance_m=max_distance,max_fault_stop_after_departure_s=max_stop_time,
        independently_required_distance_m=length,independently_total_horizon_s=total+tau)


def audit(result,observation,selection,selected,source):
    bounds=result['envelope']['volume']
    volume=QueryVolume(bounds['name'],bounds['world_frame'],tuple(bounds['lower']),tuple(bounds['upper']))
    geometry=independent(result['geometry'],volume,observation,selection['frames'][0]['pose'],selected['pose'],
                         source['mapped']['observation']['projection'])
    return dict(motion=independent_stop(result),geometry=geometry)


def page(report):
    def table(rows):
        lines=[]
        for name,r in rows:
            g=r['geometry']; e=r['envelope']
            lines.append(f'<tr><td>{html.escape(name)}</td><td>{LABELS[g["status"]]}</td><td>{e["required_distance_m"]:.3f}</td><td>{r["remaining_evidence_s"]*1000:.1f} ms</td><td>{r["stopping_horizon_s"]:.3f} s</td><td>{"会过期" if r["expires_before_stop"] else "期限满足"}</td></tr>')
        return '<div class="scroll"><table><tr><th>候选</th><th>深度几何</th><th>需覆盖前进距离 / m</th><th>证据剩余</th><th>覆盖到停止需时</th><th>停止前期限</th></tr>'+''.join(lines)+'</table></div>'
    frames=[]; options=[]
    for i,row in enumerate(report['frames']):
        p=row['planning']; rows=[(str(a['next_cell']),a['result']) for a in p['alternatives']]
        options.append(f'<option value="{i}">{row["offset_s"]} 秒图像</option>')
        details=html.escape(json.dumps(p,ensure_ascii=False,indent=2))
        frames.append(f'<section class="frame" data-frame="{i}" style="display:{"block" if i==0 else "none"}"><h2>{row["offset_s"]} 秒：四个候选第一步</h2><div class="intro"><img src="{row["image"]}" alt="该时刻原始相机图像"><div><p>假定格子空闲的 A* 路线：<strong>(1,1) → (1,2)</strong></p><p>加入机体、深度和停止时效后：<strong>没有可执行候选</strong></p><p>参考坐标中的假设起点为 (0,0,1.75) 米。该点不是相机实际位置或真实无人机位置。</p></div></div>{table(rows)}<details><summary>查看全部候选、停止范围与拒绝原因</summary><pre>{details}</pre></details></section>')
    variants=table([(v['name'],v['result']) for v in report['variants']])
    s=report['summary']
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>机体与停止时效的候选规划</title><style>body{margin:0;padding:24px;background:#eef2f6;color:#193347;font:16px/1.7 system-ui}main{max-width:1200px;margin:auto}section{background:white;padding:22px;border-radius:12px;margin:20px 0}h1{font-size:30px}h2{font-size:22px}select{font:inherit;padding:9px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #dde3eb;white-space:nowrap}.scroll{overflow:auto}.intro{display:grid;grid-template-columns:minmax(240px,420px) 1fr;gap:24px}.intro img{width:100%;border-radius:8px}strong{color:#8a401d}pre{overflow:auto;max-height:430px;font-size:12px}a{color:#175b96}@media(max-width:700px){.intro{grid-template-columns:1fr}}</style><main><h1>从“现在能看见”到“停止前仍有依据”</h1>
<section><p>六帧 × 四方向，以及首帧三个条件对照；共 '''+str(s['candidates'])+''' 个候选。全部是离线假设运动，没有执行飞行。</p><p>原始 A* 对照假定全部格空闲且机体是点，这个假设不是传感器结论。新增检查覆盖整段机体、定位误差和故障停止范围；观测不能只在出发时有效。</p><label for="frame">图像时刻：</label><select id="frame">'''+''.join(options)+'''</select></section>'''+''.join(frames)+'''<section><h2>首帧三项条件对照</h2>'''+variants+'''<p>“近似点机体”人为移除了大部分余量，只用来显示取舍，不是正常机体的可行方案。降低速度会减小反应距离，也会延长整段用时。</p><p>以相机位置为机体中心时，起始机体部分区域在镜头后方；不能裁掉它来获得完整视野。</p></section><section><h2>结果与限制</h2><p>全部候选均在最迟停止时刻前超过原观测期限。处理结果及时交付不等于有足够的运动时间。</p><p>几何检查仍有表面、缺测或视野问题；即使期限满足，普通点采样也不证明整段空闲，运动参数没有经过实机标定。此页不是避障成功率或实时控制成绩。</p><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定方案</a> · <a href="dataset-sources.json">TUM 署名与许可</a></p></section></main><script>document.getElementById('frame').addEventListener('change',function(){document.querySelectorAll('.frame').forEach(e=>e.style.display=e.dataset.frame===this.value?'block':'none');});</script></html>'''


def run(dataset,converted_dir,parent_dir,output,verify=False):
    check_manifest(dataset); check_manifest(converted_dir); check_manifest(parent_dir); verify_dataset(dataset)
    converted=json.loads((converted_dir/'report.json').read_text(encoding='utf-8'))
    parent=json.loads((parent_dir/'report.json').read_text(encoding='utf-8'))
    selection=json.loads((dataset/'selection.json').read_text(encoding='utf-8'))
    if parent['converted_sha256']!=sha(converted_dir/'report.json') or parent['dataset_sha256']!=sha(dataset/'manifest.json'):
        raise ValueError('source archive ancestry differs')
    if len(converted['frames'])!=6 or [r['offset_s'] for r in selection['frames']]!=[2,4,6,8,10,12]:
        raise ValueError('fixed frame selection differs')
    transform=reference(converted); fixed=sources(parent)
    report=dict(sources=fixed,parent_sha256=sha(parent_dir/'report.json'),converted_sha256=sha(converted_dir/'report.json'),
        dataset_sha256=sha(dataset/'manifest.json'),timing_scope='offline_saved_conversion_completion_times',
        new_model_calls=0,physical_flights=0,frames=[],variants=[])
    for i,(selected,source) in enumerate(zip(selection['frames'],converted['frames'])):
        observation=load_packed(dataset,selected,selection['origin_s'])
        inspector=DepthVolumeInspector(observation,source['finished'],transform,camera_id='tum-freiburg3-rgb',clock_id=CLOCK,expected_intrinsics=INTRINSICS)
        now=source['finished']['now_s']
        comparison=planning_comparison(inspector,origin=ORIGIN,step_m=.5,start=(1,1),goal=(1,2),config=MotionConfig(),now_s=now,clock_id=CLOCK,error_bound_m=.02)
        if len(comparison['alternatives'])!=4 or comparison['baseline_route']!=[(1,1),(1,2)]: raise ValueError('fixed planning topology differs')
        audits=[audit(r['result'],observation,selection,selected,source) for r in comparison['alternatives']]
        report['frames'].append(dict(offset_s=selected['offset_s'],image=f'frame-{i}.png',image_sha256=sha(dataset/selected['rgb'][1]),planning=comparison,audits=audits))
        if i==0:
            forward=comparison['alternatives'][0]['result']['envelope']['candidate']
            candidate=RestMotionCandidate(forward['name'],forward['world_frame'],forward['reference_fingerprint'],tuple(forward['start']),tuple(forward['end']))
            camera=inspector.pose.position; end=tuple(a+.5*d for a,d in zip(camera,inspector.pose.forward))
            camera_candidate=replace(candidate,name='相机处起步',start=camera,end=end)
            specs=[('相机处完整机体',camera_candidate,MotionConfig(),.02),
                   ('不现实的近似点机体',candidate,MotionConfig(body_radius_m=.01,reaction_s=0.,distance_margin_m=0.),0.),
                   ('降低速度至 0.1 m/s',candidate,MotionConfig(max_speed_mps=.1),.02)]
            for name,c,config,error in specs:
                result=evaluate_motion(inspector,c,config,now_s=now,clock_id=CLOCK,error_bound_m=error)
                report['variants'].append(dict(name=name,result=result,audit=audit(result,observation,selection,selected,source)))
        print('checked',selected['offset_s'],[(a['next_cell'],a['result']['geometry']['status'],a['result']['expires_before_stop']) for a in comparison['alternatives']],flush=True)
    results=[a['result'] for row in report['frames'] for a in row['planning']['alternatives']]+[v['result'] for v in report['variants']]
    audits=[a for row in report['frames'] for a in row['audits']]+[v['audit'] for v in report['variants']]
    report['summary']=dict(candidates=len(results),expires_before_stop=sum(r['expires_before_stop'] for r in results),
        geometry_statuses=dict(Counter(r['geometry']['status'] for r in results)),
        remaining_evidence_s=[min(r['remaining_evidence_s'] for r in results),max(r['remaining_evidence_s'] for r in results)],
        normal_stopping_horizon_s=report['frames'][0]['planning']['alternatives'][0]['result']['stopping_horizon_s'],
        checked_pixels=sum(r['geometry']['checked_pixels'] for r in results),
        fault_times_checked=sum(a['motion']['fault_times_checked'] for a in audits),
        max_witness_difference_m=max(a['geometry']['max_witness_difference_m'] for a in audits),executed_movements=0)
    if sources(parent)!=fixed: raise ValueError('source changed')
    report=json.loads(json.dumps(report))
    if verify:
        count=check_manifest(output)
        if report!=json.loads((output/'report.json').read_text(encoding='utf-8')): raise ValueError('motion/depth replay differs')
        if page(report)!=(output/'demo.html').read_text(encoding='utf-8'): raise ValueError('page differs')
        for row in report['frames']:
            if sha(output/row['image'])!=row['image_sha256']: raise ValueError('source image differs')
        if (output/'protocol.md').read_bytes()!=(ROOT/'docs/MOTION_VOLUME_PROTOCOL.md').read_bytes(): raise ValueError('protocol differs')
        if (output/'dataset-sources.json').read_bytes()!=(dataset/'sources.json').read_bytes(): raise ValueError('attribution differs')
        if check_manifest(output)!=count: raise ValueError('archive changed')
    else:
        output.mkdir(parents=True,exist_ok=False)
        for row,selected in zip(report['frames'],selection['frames']): shutil.copyfile(dataset/selected['rgb'][1],output/row['image'])
        shutil.copyfile(ROOT/'docs/MOTION_VOLUME_PROTOCOL.md',output/'protocol.md')
        shutil.copyfile(dataset/'sources.json',output/'dataset-sources.json')
        dump(output/'report.json',report); (output/'demo.html').write_text(page(report),encoding='utf-8')
        dump(output/'manifest.json',manifest(output)); count=check_manifest(output)
    return dict(verified=verify,files=count,sources=len(fixed),**report['summary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=ROOT/'work/tum-rgbd-bag-input-01')
    parser.add_argument('--converted',type=Path,default=ROOT/'work/frame-transform-01')
    parser.add_argument('--parent',type=Path,default=ROOT/'work/depth-volume-01')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.converted,args.parent,args.output,args.verify),ensure_ascii=False,indent=2),flush=True)
