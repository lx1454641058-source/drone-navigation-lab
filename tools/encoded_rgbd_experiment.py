"""来源：本项目原创。相邻交替对照 PNG 重新编码与验证后直接存档。"""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.realvision import dump,sha
from drone_nav.encoded_rgbd import load_encoded,EncodedVisionSession
from drone_nav.packed_rgbd import load_packed,prepare_packed,handoff_decision
from drone_nav.tum_rgbd import load_observation,project_record,CLOCK
from tools.packed_rgbd_experiment import schedule as paired_schedule
from tools.vision_runtime_experiment import check_timing,equivalence,tensor_bytes
from tools.tum_rgbd_experiment import check_manifest,manifest,independent_projection
from tools.tinyformer_probe import prepare,ROOT as MODEL_ROOT
from tools.tum_bag_subset import verify_dataset

OWN=('drone_nav/encoded_rgbd.py','tools/encoded_rgbd_experiment.py',
     'tests/test_encoded_rgbd.py','docs/ENCODED_RGBD_PROTOCOL.md')


def schedule():
    return [dict(repeat=r['repeat'],index=r['index'],
                 arm='reencode' if r['arm']=='reference' else 'encoded') for r in paired_schedule()]


def sources(parent):
    for name,expected in parent['sources'].items():
        if sha(ROOT/name)!=expected: raise ValueError('frozen source differs: '+name)
    return dict(parent['sources'],**{name:sha(ROOT/name) for name in OWN})


def summarize(frames):
    result={}
    for arm in ('reencode','encoded'):
        rows=[r for r in frames if r['arm']==arm]
        total=[r['total_until_handoff_s'] for r in rows]
        result[arm]=dict(calls=len(rows),median_s=statistics.median(total),min_s=min(total),max_s=max(total),
            timely=sum(not r['handoff']['expired_at_handoff'] for r in rows),
            equivalent=sum(all(r['equivalence'].values()) for r in rows),
            median_load_s=statistics.median(r['load_s'] for r in rows),
            median_png_s=statistics.median(r['call']['python_stages_s']['png'] for r in rows),
            median_model_call_s=statistics.median(r['model_call_s'] for r in rows),
            median_inference_ms=statistics.median(r['call']['result']['inference_ms'] for r in rows),
            median_projection_s=statistics.median(r['projection_s'] for r in rows),
            samples_before_handoff=sum(len(o['samples']) for r in rows for o in r['projection']['projection']['observations']),
            samples_timely_at_handoff=sum(len(o['samples']) for r in rows if not r['handoff']['expired_at_handoff']
                                         for o in r['projection']['projection']['observations']))
    return result


def page(report):
    names={'reencode':'重新编码 PNG','encoded':'验证后保存原始 PNG'}
    cards=[]
    for arm,s in report['summary'].items():
        cards.append(f'<article><h2>{names[arm]}</h2><strong>{s["median_s"]*1000:.1f} ms</strong>'
                     f'<p>处理合计中位数 · 时效满足 {s["timely"]}/{s["calls"]}<br>'
                     f'范围 {s["min_s"]*1000:.1f}～{s["max_s"]*1000:.1f} ms<br>'
                     f'PNG 存档中位数 {s["median_png_s"]*1000:.1f} ms<br>'
                     f'加载中位数 {s["median_load_s"]*1000:.1f} ms<br>'
                     f'输入及模型输出一致 {s["equivalent"]}/{s["calls"]}</p></article>')
    rows=[]
    for r in report['frames']:
        rows.append(f'<tr><td>{r["repeat"]+1}</td><td>{r["offset_s"]} s</td><td>{names[r["arm"]]}</td>'
                    f'<td>{r["load_s"]*1000:.1f}</td><td>{r["call"]["python_stages_s"]["png"]*1000:.1f}</td>'
                    f'<td>{r["model_call_s"]*1000:.1f}</td><td>{r["projection_s"]*1000:.1f}</td>'
                    f'<td>{r["total_until_handoff_s"]*1000:.1f}</td><td>{r["handoff"]["reason"]}</td></tr>')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PNG 存档与真实视觉处理耗时</title><style>body{{font:16px/1.7 system-ui;background:#edf2f5;color:#182e40;margin:0;padding:24px}}main{{max-width:1240px;margin:auto}}section,article{{background:white;padding:22px;border-radius:12px;margin:18px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}}strong{{font-size:30px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dbe2e8;white-space:nowrap;text-align:left}}.scroll{{overflow:auto}}.warn{{color:#86441e}}a{{color:#125c9a}}</style>
<main><h1>保留相同图像，减少 PNG 存档耗时</h1><p>16 线程 · 原始 float32 张量 · 同一模型会话 · 六张图逐对交替，重复两轮</p>
<div class="cards">{''.join(cards)}</div><section><h2>这次改变了什么</h2><p>原路径重新编码 PNG；新路径在所有原加载检查之外，额外读取并解码原 PNG，核对格式、尺寸、模式和每个像素，再将已绑定的不可变字节直接存档。额外验证计入加载耗时，输入与模型输出逐字节核对。</p>
<p class="warn">本页属于离线开发对照；时效满足不等于真实视觉可靠或允许飞行。TUM 为室内手持相机和动捕参考位姿，仍未对齐导航坐标，仍有漏检、误检及背景点。</p>
<p>合计从加载到空间检查结束，之后再次检查原 0.5 秒有效期，包含配套观测到达差。模型初始化单列；最终时效判断、报告生成、真实传输、调度和控制未计入。未清空系统文件缓存，也未后台存档。</p>
<p><a href="report.json">完整报告</a> · <a href="protocol.md">固定方案</a> · <a href="dataset-sources.json">数据来源</a></p></section>
<section><h2>24 次逐帧记录（毫秒）</h2><p>PNG 存档已包含在“模型及存档”内，不能重复相加。</p><div class="scroll"><table><thead><tr><th>轮次</th><th>图像</th><th>路径</th><th>加载</th><th>PNG 存档</th><th>模型及存档</th><th>空间检查</th><th>合计</th><th>交付判断</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>状态解释</h2><p>VISUAL_TARGET_HOLD：有候选目标，拒绝继续；VISUAL_CONTEXT_HOLD：时间或空间上下文不满足，拒绝继续；NO_DETECTION_REQUIRES_GEOMETRY：空检测仍需几何检查，不证明空闲。</p><p>TUM 数据署名：J. Sturm 等，IROS 2012，CC BY 4.0；模型 LICENSE/NOTICE 在归档 model-source 中保留。</p></section></main></html>'''


def run(dataset,baseline_dir,parent_dir,output,verify=False):
    from PIL import Image
    for directory in (dataset,baseline_dir,parent_dir): check_manifest(directory)
    verify_dataset(dataset)
    selection=json.loads((dataset/'selection.json').read_text(encoding='utf-8'))
    baseline=json.loads((baseline_dir/'report.json').read_text(encoding='utf-8'))
    parent=json.loads((parent_dir/'report.json').read_text(encoding='utf-8'))
    fixed=dict(sources=sources(parent),baseline_sha256=sha(baseline_dir/'report.json'),
               parent_sha256=sha(parent_dir/'report.json'),dataset_sha256=sha(dataset/'manifest.json'))
    if verify:
        count=check_manifest(output); report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if any(report[k]!=v for k,v in fixed.items()): raise ValueError('fixed source or input differs')
        if [{k:r[k] for k in ('repeat','index','arm')} for r in report['frames']]!=schedule():
            raise ValueError('fixed order differs')
        if (output/'protocol.md').read_bytes()!=(ROOT/'docs/ENCODED_RGBD_PROTOCOL.md').read_bytes():
            raise ValueError('protocol differs')
        if (output/'sensor/runtime-lock.json').read_bytes()!=(baseline_dir/'sensor/runtime-lock.json').read_bytes():
            raise ValueError('runtime asset lock differs')
        points=0; independent_timely={arm:0 for arm in ('reencode','encoded')}; crossings=0
        with EncodedVisionSession(output/'sensor',threads=16,input_mode='raw',replay=True) as detector:
            for call_index,row in enumerate(report['frames']):
                selected=selection['frames'][row['index']]
                observation,encoded=load_encoded(dataset,selected,selection['origin_s'])
                old=load_observation(dataset,selected,selection['origin_s'])
                if (observation.frame.rgb_bytes!=bytes(v for pixel in old.frame.rgb for v in pixel)
                        or observation.frame.depth_z_m!=old.frame.depth_z_m or observation.frame.pose!=old.frame.pose):
                    raise ValueError('original RGB/depth/pose differs')
                record=row['call']; directory=output/'sensor'/record['directory']
                if (row['threads']!=16 or row['input_mode']!='raw' or row['directory']!='sensor'
                        or row['offset_s']!=selected['offset_s'] or record['index']!=call_index
                        or record['threads']!=16 or record['input_mode']!='raw'):
                    raise ValueError('call identity differs')
                if (row['arm']=='encoded')!=(record.get('image_archive_mode')=='validated-source-png'):
                    raise ValueError('PNG archive method differs')
                if json.loads((directory/'call.json').read_text(encoding='utf-8'))!=record or json.loads((directory/'result.json').read_text(encoding='utf-8'))!=record['result']:
                    raise ValueError('call/result record differs')
                if (directory/'input.png').read_bytes()!=encoded.png_bytes: raise ValueError('original PNG archive differs')
                with Image.open(directory/'input.png') as image:
                    expected=prepare(image,record['window'])
                    if expected!=prepare_packed(image,record['window']) or expected!=tensor_bytes(directory,'raw'):
                        raise ValueError('tensor differs')
                if detector.replay_frame(record)['boxes']!=record['result']['boxes']: raise ValueError('decoded boxes differ')
                matched=equivalence(output,baseline_dir,row,baseline['frames'][row['index']])
                if matched!=row['equivalence'] or not all(matched.values()): raise ValueError('frozen output differs')
                if record['input_rgb_sha256']!=sha(directory/'input.png') or record['model_sha256']!=baseline['frames'][row['index']]['call']['model_sha256']:
                    raise ValueError('model/input digest differs')
                check_timing(row)
                ready=max(observation.captured_at_s,observation.depth_at_s,observation.pose_at_s)
                completed=ready+row['load_s']+row['model_call_s']
                projected=project_record(old,record,completed_at_s=completed,now_s=completed)
                if projected!=row['projection']: raise ValueError('spatial replay differs')
                handoff=handoff_decision(projected,consumer_at_s=ready+row['total_until_handoff_s'],clock_id=CLOCK)
                if handoff!=row['handoff']: raise ValueError('handoff differs')
                # Independently calculate the age from input times, without using
                # projected validity fields or the handoff implementation.
                origin=selection['origin_s']; capture=selected['rgb'][0]-origin
                input_ready=max(selected[k][0]-origin for k in ('rgb','depth','pose'))
                age=input_ready-capture+row['load_s']+row['model_call_s']+row['projection_s']
                expired=age>.5+1e-9
                reason=('VISUAL_CONTEXT_HOLD' if expired else 'VISUAL_TARGET_HOLD'
                        if record['result']['boxes'] else 'NO_DETECTION_REQUIRES_GEOMETRY')
                if expired!=handoff['expired_at_handoff'] or reason!=handoff['reason'] or handoff['flight_authorized'] or handoff['navigation_frame_alignment_available']:
                    raise ValueError('independent age/decision differs')
                independent_timely[row['arm']]+=not expired
                crossings+=expired and completed-capture<=.5+1e-9
                points+=independent_projection(old,projected,selected['pose'])
                print('verified',row['repeat'],row['offset_s'],row['arm'],flush=True)
        if summarize(report['frames'])!=report['summary'] or page(report)!=(output/'demo.html').read_text(encoding='utf-8'):
            raise ValueError('summary or page differs')
        if report['session_exit_code']!=0 or report['startup_s']<=0: raise ValueError('session lifecycle differs')
        if check_manifest(output)!=count: raise ValueError('replay altered archive')
        return dict(verified=True,calls=24,files=count,sources=len(fixed['sources']),independent_samples=points,
                    independent_timely_calls=independent_timely,expired_during_projection=crossings)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(ROOT/'docs/ENCODED_RGBD_PROTOCOL.md',output/'protocol.md')
    shutil.copyfile(dataset/'sources.json',output/'dataset-sources.json')
    (output/'model-source').mkdir()
    for name in ('LICENSE','NOTICE','sources.json'): shutil.copyfile(MODEL_ROOT/name,output/'model-source'/name)
    report=dict(**fixed,frames=[])
    with EncodedVisionSession(output/'sensor',threads=16,input_mode='raw') as detector:
        report['startup_s']=detector.startup_s
        for item in schedule():
            selected=selection['frames'][item['index']]
            start=perf_counter()
            if item['arm']=='encoded': observation,encoded=load_encoded(dataset,selected,selection['origin_s'])
            else: observation=load_packed(dataset,selected,selection['origin_s'])
            loaded=perf_counter()
            record=detector.detect_encoded(encoded) if item['arm']=='encoded' else detector.detect_packed(observation.frame)
            modeled=perf_counter(); load_s=loaded-start; call_s=modeled-loaded
            ready=max(observation.captured_at_s,observation.depth_at_s,observation.pose_at_s)
            completed=ready+load_s+call_s
            projected=project_record(observation,record,completed_at_s=completed,now_s=completed)
            checked=perf_counter(); total=checked-start
            handoff=handoff_decision(projected,consumer_at_s=ready+total,clock_id=CLOCK)
            row=dict(**item,directory='sensor',threads=16,input_mode='raw',offset_s=selected['offset_s'],
                     call=record,load_s=load_s,model_call_s=call_s,projection_s=checked-modeled,
                     total_until_handoff_s=total,projection=projected,handoff=handoff)
            row['equivalence']=equivalence(output,baseline_dir,row,baseline['frames'][item['index']])
            check_timing(row); report['frames'].append(row)
            print('ran',item['repeat'],selected['offset_s'],item['arm'],round(total,4),handoff['reason'],
                  'equivalent',all(row['equivalence'].values()),flush=True)
    report['session_exit_code']=detector.process.returncode
    if sources(parent)!=fixed['sources']: raise ValueError('sources changed during run')
    report['summary']=summarize(report['frames'])
    dump(output/'report.json',report); (output/'demo.html').write_text(page(report),encoding='utf-8')
    dump(output/'manifest.json',manifest(output))
    return report['summary']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=ROOT/'work/tum-rgbd-bag-input-01')
    parser.add_argument('--baseline',type=Path,default=ROOT/'work/tum-rgbd-01')
    parser.add_argument('--parent',type=Path,default=ROOT/'work/vision-runtime-01')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.baseline,args.parent,args.output,args.verify),indent=2),flush=True)
