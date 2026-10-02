"""来源：本项目原创。固定 CPU 配置的完整视觉处理对照及只读重放。"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.realvision import dump,sha
from drone_nav.vision_runtime import CPUVisionSession
from drone_nav.packed_rgbd import load_packed,prepare_packed,handoff_decision
from drone_nav.tum_rgbd import load_observation,project_record,CLOCK
from tools.tinyformer_probe import prepare,ROOT as MODEL_ROOT
from tools.tum_rgbd_experiment import check_manifest,manifest,independent_projection
from tools.tum_bag_subset import verify_dataset

CONFIGS=((4,'gzip'),(8,'gzip'),(8,'raw'),(16,'raw'))
OWN=('drone_nav/vision_runtime.py','tools/vision_runtime_worker.cjs',
     'tests/test_vision_runtime.py','tests/test_vision_runtime_experiment.py',
     'tools/vision_runtime_experiment.py','docs/VISION_RUNTIME_PROTOCOL.md')


def blocks():
    return [dict(repeat=repeat,threads=threads,input_mode=mode,
                 directory=f'r{repeat}-t{threads}-{mode}')
            for repeat in range(2)
            for threads,mode in (CONFIGS if repeat==0 else CONFIGS[::-1])]


def schedule():
    return [dict(**block,index=index) for block in blocks() for index in range(6)]


def sources(parent):
    for name,expected in parent['sources'].items():
        if sha(ROOT/name)!=expected: raise ValueError('frozen source differs: '+name)
    return dict(parent['sources'],**{name:sha(ROOT/name) for name in OWN})


def tensor_bytes(directory,mode):
    if mode=='gzip': return gzip.decompress((directory/'input.gz').read_bytes())
    if mode=='raw': return (directory/'input.f32').read_bytes()
    raise ValueError('unknown tensor format')


def equivalence(output,reference,row,baseline_row):
    current=output/row['directory']/row['call']['directory']
    previous=reference/'sensor'/baseline_row['call']['directory']
    result=dict(input_png=sha(current/'input.png')==sha(previous/'input.png'),
                input_bytes=tensor_bytes(current,row['input_mode'])==tensor_bytes(previous,'gzip'),
                decoded_boxes=row['call']['result']['boxes']==baseline_row['call']['result']['boxes'])
    for name in ('logits','boxes'):
        result[name+'_bytes']=gzip.decompress((current/(name+'.gz')).read_bytes())==gzip.decompress((previous/(name+'.gz')).read_bytes())
    return result


def check_timing(row):
    call=row['call']; python=call['python_stages_s']; worker=call['result']['worker_stages_ms']
    values=[row[k] for k in ('load_s','model_call_s','projection_s','total_until_handoff_s')]
    values+=[call['elapsed_s'],call['prepare_s'],call['input_archive_s'],*python.values(),*worker.values()]
    if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError('invalid stage duration')
    def equal(a,b):
        if abs(a-b)>1e-7: raise ValueError('timing sum differs')
    equal(row['load_s']+row['model_call_s']+row['projection_s'],row['total_until_handoff_s'])
    equal(sum(python.values()),call['elapsed_s'])
    equal(python['png']+python['tensor'],call['prepare_s'])
    equal(python['input_archive'],call['input_archive_s'])
    equal(worker['inference'],call['result']['inference_ms'])
    if call['elapsed_s']>row['model_call_s']+1e-7 or sum(worker.values())/1000>python['worker_roundtrip']+1e-7:
        raise ValueError('nested duration exceeds parent')


def summarize(frames):
    result={}
    for threads,mode in CONFIGS:
        rows=[r for r in frames if (r['threads'],r['input_mode'])==(threads,mode)]
        durations=[r['total_until_handoff_s'] for r in rows]
        if not rows: raise ValueError('missing configuration')
        result[f't{threads}-{mode}']=dict(calls=len(rows),median_s=statistics.median(durations),
            min_s=min(durations),max_s=max(durations),
            timely=sum(not r['handoff']['expired_at_handoff'] for r in rows),
            equivalent=sum(all(r['equivalence'].values()) for r in rows),
            reasons={reason:sum(r['handoff']['reason']==reason for r in rows)
                     for reason in sorted({r['handoff']['reason'] for r in rows})},
            median_load_s=statistics.median(r['load_s'] for r in rows),
            median_model_call_s=statistics.median(r['model_call_s'] for r in rows),
            median_projection_s=statistics.median(r['projection_s'] for r in rows),
            median_python_stages_s={key:statistics.median(r['call']['python_stages_s'][key] for r in rows)
                                    for key in rows[0]['call']['python_stages_s']},
            median_worker_stages_ms={key:statistics.median(r['call']['result']['worker_stages_ms'][key] for r in rows)
                                     for key in rows[0]['call']['result']['worker_stages_ms']})
    return result


def page(report):
    cards=[]
    for label,s in report['summary'].items():
        cards.append(f'<article><h2>{label}</h2><strong>{s["median_s"]*1000:.1f} ms</strong>'
                     f'<p>合计中位数 · 时效满足 {s["timely"]}/{s["calls"]}<br>'
                     f'范围 {s["min_s"]*1000:.1f}～{s["max_s"]*1000:.1f} ms<br>'
                     f'输入和模型输出一致 {s["equivalent"]}/{s["calls"]}</p></article>')
    rows=[]
    for r in report['frames']:
        rows.append(f'<tr><td>{r["repeat"]+1}</td><td>t{r["threads"]}-{r["input_mode"]}</td>'
                    f'<td>{r["offset_s"]} s</td><td>{r["load_s"]*1000:.1f}</td>'
                    f'<td>{r["model_call_s"]*1000:.1f}</td><td>{r["projection_s"]*1000:.1f}</td>'
                    f'<td>{r["total_until_handoff_s"]*1000:.1f}</td>'
                    f'<td>{"有效" if not r["handoff"]["expired_at_handoff"] else "过期"}</td>'
                    f'<td>{r["handoff"]["reason"]}</td></tr>')
    stages=[]
    for label,s in report['summary'].items():
        p=s['median_python_stages_s']; w=s['median_worker_stages_ms']
        stages.append(f'<tr><td>{label}</td><td>{p["png"]*1000:.1f}</td><td>{p["tensor"]*1000:.1f}</td>'
                      f'<td>{p["input_archive"]*1000:.1f}</td><td>{w["read_input"]:.1f}</td>'
                      f'<td>{w["validate_input"]:.1f}</td><td>{w["inference"]:.1f}</td>'
                      f'<td>{w["decode"]:.1f}</td><td>{w["archive_outputs"]:.1f}</td></tr>')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPU 视觉处理完整流程对照</title><style>body{{font:16px/1.65 system-ui;background:#edf2f5;color:#172b3e;margin:0;padding:24px}}main{{max-width:1280px;margin:auto}}section,article{{background:white;border-radius:12px;padding:20px;margin-bottom:18px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}strong{{font-size:30px}}h2{{font-size:20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d9e1e8;white-space:nowrap;text-align:left}}.scroll{{overflow:auto}}.warn{{color:#8a431a}}a{{color:#155ea1}}</style>
<main><h1>线程与存档格式：完整视觉处理流程对照</h1><p>相同六张真实 RGB-D 图像 · 四配置 × 两轮 · 顺序反转 · 一次只运行一个模型会话</p>
<div class="cards">{''.join(cards)}</div><section><h2>结果应如何理解</h2>
<p>合计包含读取图像/深度/位姿、PNG 与张量准备、输入存档、模型推理与输出存档、空间检查；空间检查后重新核对原 0.5 秒有效期。配套观测到达时间也计入年龄。</p>
<p class="warn">时效满足不代表可以飞行。TUM 数据来自室内手持相机，使用动捕参考位姿，尚未对齐导航坐标；仍有误检、漏检及框内背景点。本页没有无人机飞行，也不报告识别准确率。</p>
<p>这是本机离线开发实验，不清空操作系统文件缓存；模型初始化单列，最终时效判断、报告写入、相机传输、在线调度与控制未计入。没有修改模型、像素、阈值或时效。</p>
<p><a href="report.json">完整报告</a> · <a href="protocol.md">固定实验方案</a> · <a href="dataset-sources.json">数据署名与许可</a></p></section>
<section><h2>各阶段中位数（毫秒）</h2><p>各阶段中位数不能直接相加当作总耗时；总耗时按每次实际记录计算。</p><div class="scroll"><table><thead><tr><th>配置</th><th>PNG</th><th>张量</th><th>输入存档</th><th>读取张量</th><th>验证张量</th><th>模型推理</th><th>解码</th><th>输出存档</th></tr></thead><tbody>{''.join(stages)}</tbody></table></div></section>
<section><h2>48 次调用，逐次保留（毫秒）</h2><div class="scroll"><table><thead><tr><th>轮次</th><th>配置</th><th>时刻</th><th>加载</th><th>模型及存档</th><th>空间检查</th><th>合计</th><th>时效</th><th>交付判断</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>判断含义</h2><p>VISUAL_CONTEXT_HOLD：时间或空间上下文不满足，拒绝继续。VISUAL_TARGET_HOLD：有候选目标，拒绝继续。NO_DETECTION_REQUIRES_GEOMETRY：空检测仍需几何检查，不证明空闲。</p><p>原始 TUM 数据：J. Sturm 等，IROS 2012，CC BY 4.0。模型来源和许可证在 model-source 中保留。</p></section></main></html>'''


def run(dataset,reference,parent_dir,output,verify=False):
    from PIL import Image
    for directory in (dataset,reference,parent_dir): check_manifest(directory)
    verify_dataset(dataset)
    selected=json.loads((dataset/'selection.json').read_text(encoding='utf-8'))
    baseline=json.loads((reference/'report.json').read_text(encoding='utf-8'))
    parent=json.loads((parent_dir/'report.json').read_text(encoding='utf-8'))
    fixed=dict(sources=sources(parent),baseline_sha256=sha(reference/'report.json'),
               parent_sha256=sha(parent_dir/'report.json'),dataset_sha256=sha(dataset/'manifest.json'))
    if verify:
        count=check_manifest(output)
        report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if any(report[k]!=v for k,v in fixed.items()): raise ValueError('source/input identity differs')
        if [{k:r[k] for k in ('repeat','threads','input_mode','directory','index')} for r in report['frames']]!=schedule():
            raise ValueError('fixed call schedule differs')
        if [{k:r[k] for k in ('repeat','threads','input_mode','directory')} for r in report['sessions']]!=blocks():
            raise ValueError('session schedule differs')
        points=0
        if (output/'protocol.md').read_bytes()!=(ROOT/'docs/VISION_RUNTIME_PROTOCOL.md').read_bytes():
            raise ValueError('saved protocol differs')
        for block,session_record in zip(blocks(),report['sessions']):
            folder=output/block['directory']
            if json.loads((folder/'session.json').read_text(encoding='utf-8'))!=session_record:
                raise ValueError('session record differs')
            if session_record['exit_code']!=0 or not math.isfinite(session_record['startup_s']) or session_record['startup_s']<0:
                raise ValueError('invalid session lifecycle')
            if (folder/'runtime-lock.json').read_bytes()!=(reference/'sensor/runtime-lock.json').read_bytes():
                raise ValueError('runtime assets differ')
            with CPUVisionSession(folder,threads=block['threads'],input_mode=block['input_mode'],replay=True) as detector:
                for row in [r for r in report['frames'] if r['directory']==block['directory']]:
                    choice=selected['frames'][row['index']]
                    observation=load_observation(dataset,choice,selected['origin_s'])
                    packed=load_packed(dataset,choice,selected['origin_s'])
                    if (packed.frame.depth_z_m!=observation.frame.depth_z_m or packed.frame.pose!=observation.frame.pose
                            or packed.frame.rgb_bytes!=bytes(v for pixel in observation.frame.rgb for v in pixel)):
                        raise ValueError('packed observation differs')
                    record=row['call']; directory=folder/record['directory']
                    if record['index']!=row['index'] or record['threads']!=row['threads'] or record['input_mode']!=row['input_mode'] or row['offset_s']!=choice['offset_s']:
                        raise ValueError('call identity differs')
                    if json.loads((directory/'call.json').read_text(encoding='utf-8'))!=record or json.loads((directory/'result.json').read_text(encoding='utf-8'))!=record['result']:
                        raise ValueError('saved call differs')
                    if detector.replay_frame(record)['boxes']!=record['result']['boxes']: raise ValueError('decode differs')
                    with Image.open(directory/'input.png') as image:
                        expected=prepare(image,record['window'])
                        if image.tobytes()!=packed.frame.rgb_bytes or expected!=prepare_packed(image,record['window']) or expected!=tensor_bytes(directory,row['input_mode']):
                            raise ValueError('pixel/tensor differs')
                    matched=equivalence(output,reference,row,baseline['frames'][row['index']])
                    if matched!=row['equivalence'] or not all(matched.values()): raise ValueError('output differs from frozen reference')
                    if record['input_rgb_sha256']!=sha(directory/'input.png') or record['model_sha256']!=baseline['frames'][row['index']]['call']['model_sha256']:
                        raise ValueError('call digest differs')
                    check_timing(row)
                    ready=max(observation.captured_at_s,observation.depth_at_s,observation.pose_at_s)
                    completed=ready+row['load_s']+row['model_call_s']
                    projected=project_record(observation,record,completed_at_s=completed,now_s=completed)
                    if projected!=row['projection']: raise ValueError('spatial replay differs')
                    handoff=handoff_decision(projected,consumer_at_s=ready+row['total_until_handoff_s'],clock_id=CLOCK)
                    if handoff!=row['handoff']: raise ValueError('handoff replay differs')
                    points+=independent_projection(observation,projected,choice['pose'])
                    print('verified',block['directory'],row['offset_s'],flush=True)
        if summarize(report['frames'])!=report['summary'] or page(report)!=(output/'demo.html').read_text(encoding='utf-8'):
            raise ValueError('summary/page differs')
        if check_manifest(output)!=count: raise ValueError('replay modified archive')
        return dict(verified=True,calls=48,files=count,sources=len(fixed['sources']),independent_samples=points)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(ROOT/'docs/VISION_RUNTIME_PROTOCOL.md',output/'protocol.md')
    shutil.copyfile(dataset/'sources.json',output/'dataset-sources.json')
    (output/'model-source').mkdir()
    for name in ('LICENSE','NOTICE','sources.json'): shutil.copyfile(MODEL_ROOT/name,output/'model-source'/name)
    report=dict(**fixed,host=dict(logical_cpus=os.cpu_count(),platform=platform.platform(),python=sys.version),sessions=[],frames=[])
    for block in blocks():
        with CPUVisionSession(output/block['directory'],threads=block['threads'],input_mode=block['input_mode']) as detector:
            for index,choice in enumerate(selected['frames']):
                start=perf_counter(); observation=load_packed(dataset,choice,selected['origin_s']); loaded=perf_counter()
                record=detector.detect_packed(observation.frame); modeled=perf_counter()
                load_s=loaded-start; call_s=modeled-loaded
                ready=max(observation.captured_at_s,observation.depth_at_s,observation.pose_at_s)
                completed=ready+load_s+call_s
                projected=project_record(observation,record,completed_at_s=completed,now_s=completed)
                checked=perf_counter(); total=checked-start
                handoff=handoff_decision(projected,consumer_at_s=ready+total,clock_id=CLOCK)
                row=dict(**block,index=index,offset_s=choice['offset_s'],call=record,load_s=load_s,model_call_s=call_s,
                         projection_s=checked-modeled,total_until_handoff_s=total,projection=projected,handoff=handoff)
                row['equivalence']=equivalence(output,reference,row,baseline['frames'][index])
                check_timing(row); report['frames'].append(row)
                print('ran',block['directory'],choice['offset_s'],round(total,4),handoff['reason'],'equivalent',all(row['equivalence'].values()),flush=True)
        session_record=dict(**block,startup_s=detector.startup_s,exit_code=detector.process.returncode)
        report['sessions'].append(session_record)
        dump(output/block['directory']/'session.json',session_record)
    if sources(parent)!=fixed['sources']: raise ValueError('source changed while running')
    report['summary']=summarize(report['frames'])
    dump(output/'report.json',report)
    (output/'demo.html').write_text(page(report),encoding='utf-8')
    dump(output/'manifest.json',manifest(output))
    return report['summary']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=ROOT/'work/tum-rgbd-bag-input-01')
    parser.add_argument('--baseline',type=Path,default=ROOT/'work/tum-rgbd-01')
    parser.add_argument('--parent',type=Path,default=ROOT/'work/packed-rgbd-01')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.baseline,args.parent,args.output,args.verify),indent=2),flush=True)
