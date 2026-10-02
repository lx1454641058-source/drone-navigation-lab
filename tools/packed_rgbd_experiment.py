"""来源：本项目原创。相同图像/模型的交替耗时对照及逐字节重放。"""
import argparse
import gzip
import html
import json
from pathlib import Path
import shutil
import statistics
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.realvision import dump,sha
from drone_nav.packed_rgbd import PackedTinyFormerSession,load_packed,handoff_decision,prepare_packed
from drone_nav.tum_rgbd import load_observation,project_record,CLOCK
from tools.tinyformer_probe import prepare,ROOT as MODEL_ROOT
from tools.tum_rgbd_experiment import check_manifest,manifest,independent_projection
from tools.tum_bag_subset import verify_dataset

OWN=('drone_nav/packed_rgbd.py','tests/test_packed_rgbd.py',
     'tools/packed_rgbd_experiment.py','docs/PACKED_RGBD_PROTOCOL.md')


def schedule():
    result=[]
    for repeat in range(2):
        for index in range(6):
            arms=('reference','packed') if (repeat+index)%2==0 else ('packed','reference')
            result.extend(dict(repeat=repeat,index=index,arm=arm) for arm in arms)
    return result


def check_sources(baseline):
    for name,expected in baseline['sources'].items():
        if sha(ROOT/name)!=expected: raise ValueError('baseline source differs: '+name)
    return dict(baseline['sources'],**{name:sha(ROOT/name) for name in OWN})


def equivalence(directory,reference,row,baseline_row):
    current=directory/'sensor'/row['call']['directory']
    previous=reference/'sensor'/baseline_row['call']['directory']
    result={'input_png':sha(current/'input.png')==sha(previous/'input.png'),
        'decoded_boxes':row['call']['result']['boxes']==baseline_row['call']['result']['boxes']}
    for name in ('input','logits','boxes'):
        result[name+'_bytes']=gzip.decompress((current/(name+'.gz')).read_bytes())==gzip.decompress((previous/(name+'.gz')).read_bytes())
    return result


def summarize(frames):
    result={}
    for arm in ('reference','packed'):
        group=[r for r in frames if r['arm']==arm]
        durations=[r['total_until_handoff_s'] for r in group]
        result[arm]=dict(calls=len(group),median_s=statistics.median(durations),
            min_s=min(durations),max_s=max(durations),
            timely=sum(not r['handoff']['expired_at_handoff'] for r in group),
            equivalent=sum(all(r['equivalence'].values()) for r in group),
            median_load_s=statistics.median(r['load_s'] for r in group),
            median_model_call_s=statistics.median(r['model_call_s'] for r in group),
            median_projection_s=statistics.median(r['projection_s'] for r in group))
    return result


def page(report):
    summary=report['summary']; old,new=(summary[k] for k in ('reference','packed'))
    rows=[]
    for row in report['frames']:
        status='一致' if all(row['equivalence'].values()) else '有差异'
        rows.append(f'''<tr><td>{row['repeat']+1}</td><td>{row['offset_s']} 秒</td><td>{row['arm']}</td>
<td>{row['load_s']*1000:.1f}</td><td>{row['model_call_s']*1000:.1f}</td><td>{row['projection_s']*1000:.1f}</td>
<td>{row['total_until_handoff_s']*1000:.1f}</td><td>{row['handoff']['reason']}</td><td>{status}</td></tr>''')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实 RGB-D 输入耗时对照</title><style>body{{font:16px/1.7 system-ui;color:#172635;background:#eef2f6;margin:0;padding:28px}}main{{max-width:1250px;margin:auto}}section{{background:white;border-radius:12px;padding:24px;margin:20px 0}}.stats{{display:flex;gap:25px;flex-wrap:wrap}}strong{{font-size:28px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;border-bottom:1px solid #dbe3eb;padding:9px;white-space:nowrap}}.scroll{{overflow:auto}}p{{max-width:1100px}}.warn{{color:#8b4218}}a{{color:#1762ad}}</style>
<main><h1>真实 RGB-D：保持输入与模型不变，减少处理耗时</h1>
<section><div class="stats"><div>旧路径中位数<br><strong>{old['median_s']*1000:.1f} ms</strong><br>交付时有效 {old['timely']}/{old['calls']}</div>
<div>字节输入路径中位数<br><strong>{new['median_s']*1000:.1f} ms</strong><br>交付时有效 {new['timely']}/{new['calls']}</div>
<div>输入及两路模型输出一致<br><strong>{old['equivalent']+new['equivalent']}/24</strong></div></div>
<p>六张固定真实 RGB-D 图像，每张两种路径、重复两轮；交替顺序，同一个模型会话。保留原 0.5 秒时效、深度/位姿检查和检测门槛，合计包含空间检查。</p>
<p class="warn">这是本机离线热缓存开发对照。有效只代表本次时效检查，仍有识别错误、缺失深度、未对齐导航坐标；没有真实飞行或无人机控制。模型初始化与最终时效判断/报告写入单列或排除，未测相机传输及在线调度。</p>
<p>优化：不可变字节 RGB 避免重复逐元素检查；相同 float32 查表交由已安装 Pillow 执行。数据仍为 TUM 室内手持相机，模型和第三方许可不变。</p></section>
<section><h2>逐次测量（毫秒）</h2><div class="scroll"><table><thead><tr><th>轮次</th><th>图像</th><th>路径</th><th>加载</th><th>模型及存档</th><th>空间检查</th><th>合计</th><th>交付判断</th><th>字节对照</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>口径</h2><p>VISUAL_TARGET_HOLD：发现候选目标，拒绝继续；VISUAL_CONTEXT_HOLD：上下文或时效不满足；NO_DETECTION_REQUIRES_GEOMETRY：没有检测框，仍需原几何检查，不代表空地。</p><p>时效以原采集时刻计，包含等待配套深度/位姿到达的时间。投影结束后再次检查过期，不能用模型刚返回时仍有效来掩盖后续处理超时。</p>
<p>数据：J. Sturm 等，TUM RGB-D benchmark，IROS 2012，<a href="https://cvg.cit.tum.de/data/datasets/rgbd-dataset">CC BY 4.0 / 官方来源</a>。</p></section></main></html>'''


def run(dataset,baseline_dir,output,verify=False):
    from PIL import Image
    check_manifest(dataset); check_manifest(baseline_dir); verify_dataset(dataset)
    selections=json.loads((dataset/'selection.json').read_text(encoding='utf-8'))
    baseline=json.loads((baseline_dir/'report.json').read_text(encoding='utf-8'))
    sources=check_sources(baseline)
    if verify:
        count=check_manifest(output); report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if report['sources']!=sources or report['baseline_sha256']!=sha(baseline_dir/'report.json') or report['dataset_sha256']!=sha(dataset/'manifest.json'):
            raise ValueError('fixed sources or inputs differ')
        if [{k:r[k] for k in ('repeat','index','arm')} for r in report['frames']]!=schedule():
            raise ValueError('run order differs')
        point_count=0
        with PackedTinyFormerSession(output/'sensor',replay=True) as detector:
            for row in report['frames']:
                selected=selections['frames'][row['index']]
                observation=load_observation(dataset,selected,selections['origin_s'])
                packed=load_packed(dataset,selected,selections['origin_s'])
                if packed.frame.depth_z_m!=observation.frame.depth_z_m or packed.frame.pose!=observation.frame.pose:
                    raise ValueError('optimized geometry differs')
                if packed.frame.rgb_bytes!=bytes(v for pixel in observation.frame.rgb for v in pixel):
                    raise ValueError('optimized RGB differs')
                record=row['call']; directory=output/'sensor'/record['directory']
                if detector.replay_frame(record)['boxes']!=record['result']['boxes']: raise ValueError('decoded boxes differ')
                with Image.open(directory/'input.png') as image:
                    if image.tobytes()!=packed.frame.rgb_bytes: raise ValueError('archived RGB differs')
                    expected=prepare(image,record['window'])
                    if expected!=prepare_packed(image,record['window']) or expected!=gzip.decompress((directory/'input.gz').read_bytes()):
                        raise ValueError('reference and packed tensor differ')
                matched=equivalence(output,baseline_dir,row,baseline['frames'][row['index']])
                if matched!=row['equivalence'] or not all(matched.values()): raise ValueError('frozen output differs')
                ready=max(observation.captured_at_s,observation.depth_at_s,observation.pose_at_s)
                completed=ready+row['load_s']+row['model_call_s']
                projected=project_record(observation,record,completed_at_s=completed,now_s=completed)
                if projected!=row['projection']: raise ValueError('projection replay differs')
                total=row['load_s']+row['model_call_s']+row['projection_s']
                if abs(total-row['total_until_handoff_s'])>1e-8: raise ValueError('timing sum differs')
                handoff=handoff_decision(projected,consumer_at_s=ready+row['total_until_handoff_s'],clock_id=CLOCK)
                if handoff!=row['handoff']: raise ValueError('handoff replay differs')
                point_count+=independent_projection(observation,projected,selected['pose'])
                print('verified',row['repeat'],row['offset_s'],row['arm'],flush=True)
        if summarize(report['frames'])!=report['summary'] or page(report)!=(output/'demo.html').read_text(encoding='utf-8'):
            raise ValueError('summary or page differs')
        return dict(verified=True,calls=24,files=count,independent_samples=point_count)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(ROOT/'docs/PACKED_RGBD_PROTOCOL.md',output/'protocol.md')
    shutil.copyfile(dataset/'sources.json',output/'dataset-sources.json')
    (output/'model-source').mkdir()
    for name in ('LICENSE','NOTICE','sources.json'): shutil.copyfile(MODEL_ROOT/name,output/'model-source'/name)
    report=dict(sources=sources,baseline_sha256=sha(baseline_dir/'report.json'),
        dataset_sha256=sha(dataset/'manifest.json'),frames=[])
    with PackedTinyFormerSession(output/'sensor') as detector:
        report['startup_s']=detector.startup_s
        for item in schedule():
            selected=selections['frames'][item['index']]
            start=perf_counter()
            observation=(load_observation if item['arm']=='reference' else load_packed)(dataset,selected,selections['origin_s'])
            loaded=perf_counter()
            record=(detector.detect if item['arm']=='reference' else detector.detect_packed)(observation.frame)
            modeled=perf_counter()
            load_s=loaded-start; call_s=modeled-loaded
            ready=max(observation.captured_at_s,observation.depth_at_s,observation.pose_at_s)
            completed=ready+load_s+call_s
            projected=project_record(observation,record,completed_at_s=completed,now_s=completed)
            checked=perf_counter(); total=checked-start
            handoff=handoff_decision(projected,consumer_at_s=ready+total,clock_id=CLOCK)
            row=dict(**item,offset_s=selected['offset_s'],call=record,load_s=load_s,model_call_s=call_s,
                projection_s=checked-modeled,total_until_handoff_s=total,projection=projected,handoff=handoff)
            row['equivalence']=equivalence(output,baseline_dir,row,baseline['frames'][item['index']])
            report['frames'].append(row)
            print('ran',item['repeat'],selected['offset_s'],item['arm'],round(total,4),handoff['reason'],
                  'equivalent',all(row['equivalence'].values()),flush=True)
    report['summary']=summarize(report['frames'])
    dump(output/'report.json',report); (output/'demo.html').write_text(page(report),encoding='utf-8')
    dump(output/'manifest.json',manifest(output))
    return report['summary']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=Path('work/tum-rgbd-bag-input-01'))
    parser.add_argument('--baseline',type=Path,default=Path('work/tum-rgbd-01'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.baseline,args.output,args.verify),indent=2),flush=True)
