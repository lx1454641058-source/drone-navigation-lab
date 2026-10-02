"""来源：本项目原创。固定异常事件与墙钟 RGB-D 文件播放接入有界接收器。"""
import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter,sleep

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.timed_inbox import TimedObservationInbox
from drone_nav.encoded_rgbd import EncodedVisionSession,load_encoded
from drone_nav.tum_rgbd import CLOCK,WORLD,load_observation,project_record
from drone_nav.realvision import dump,sha
from tools.tum_rgbd_experiment import check_manifest,manifest,independent_projection
from tools.vision_runtime_experiment import equivalence,tensor_bytes
from tools.tum_bag_subset import verify_dataset
from tools.tinyformer_probe import prepare,ROOT as MODEL_ROOT

CAMERA='tum-freiburg3-rgb'
OWN=('drone_nav/timed_inbox.py','tests/test_timed_inbox.py',
     'tools/timed_inbox_experiment.py','docs/TIMED_INBOX_PROTOCOL.md')


def inbox():
    return TimedObservationInbox(camera_id=CAMERA,clock_id=CLOCK,world_frame=WORLD,capacity=2)


def apply(receiver,event):
    before=receiver.status()
    try:
        if event['kind']=='submit':
            answer=receiver.submit(event['result'],now_s=event['now_s'],clock_id=event.get('clock_id',CLOCK))
        elif event['kind']=='consume':
            answer=receiver.consume(now_s=event['now_s'],clock_id=event.get('clock_id',CLOCK),target_world=event['target_world'])
        else: raise ValueError('unknown event')
    except ValueError as exc:
        if receiver.status()!=before: raise ValueError('rejected event partially changed state') from exc
        answer=dict(reason='INVALID_INPUT',error=str(exc),flight_authorized=False)
    return dict(event=deepcopy(event),answer=answer,state=receiver.status())


def cases(parent):
    rows=sorted([r for r in parent['frames'] if r['arm']=='encoded' and r['repeat']==0],key=lambda r:r['index'])
    def submit(r,now=None,result=None):
        return dict(kind='submit',now_s=r['handoff']['consumer_at_s'] if now is None else now,
                    result=r['projection'] if result is None else result)
    def consume(now,world=WORLD): return dict(kind='consume',now_s=now,target_world=world)
    t=[r['handoff']['consumer_at_s'] for r in rows]
    expected=['VISUAL_TARGET_HOLD','VISUAL_TARGET_HOLD','NO_DETECTION_REQUIRES_GEOMETRY',
              'VISUAL_TARGET_HOLD','VISUAL_TARGET_HOLD','VISUAL_TARGET_HOLD']
    output=[]
    for name,delay,duplicate in [('saved-sequence',.001,False),('queued-delay',.1,False),('duplicate',.002,True)]:
        events=[]; reasons=[]
        for i,r in enumerate(rows):
            events.append(submit(r)); reasons.append('QUEUED')
            if duplicate:
                events.append(submit(r,t[i]+.001)); reasons.append('REPEATED_FRAME')
            events.append(consume(t[i]+delay)); reasons.append('EXPIRED_AT_CONSUMPTION' if delay==.1 else expected[i])
        output.append(dict(name=name,events=events,expected_reasons=reasons))
    output.extend([
        dict(name='out-of-order',events=[submit(rows[1]),submit(rows[0],t[1]+.001),consume(t[1]+.002)],
             expected_reasons=['QUEUED','OUT_OF_ORDER_CAPTURE','VISUAL_TARGET_HOLD']),
        dict(name='paused-overflow',events=[submit(rows[0]),submit(rows[1]),submit(rows[2]),consume(t[2]+.001)],
             expected_reasons=['QUEUED','QUEUED','QUEUE_OVERFLOW','STREAM_FAULT_LATCHED']),
        dict(name='wrong-target-world',events=[submit(rows[0]),consume(t[0]+.001,'east_north_up_m'),consume(t[0]+.002)],
             expected_reasons=['QUEUED','TARGET_WORLD_MISMATCH','VISUAL_TARGET_HOLD']),
    ])
    late=rows[0]['projection']['projection']['captured_at_s']+.51
    output.append(dict(name='expired-receipt',events=[submit(rows[0],late),consume(late+.001)],
                       expected_reasons=['EXPIRED_AT_RECEIPT','NO_PENDING_OBSERVATION']))
    for name,key,value,reason in [('wrong-source-world','world_frame','east_north_up_m','SOURCE_WORLD_MISMATCH'),
                                  ('wrong-source-clock','clock_id','foreign-clock','SOURCE_CLOCK_MISMATCH')]:
        changed=deepcopy(rows[0]['projection']); changed['projection'][key]=value
        output.append(dict(name=name,events=[submit(rows[0],result=changed),consume(t[0]+.001)],
                           expected_reasons=[reason,'NO_PENDING_OBSERVATION']))
    output.append(dict(name='backward-consumer-clock',events=[submit(rows[0]),consume(t[0]-.001),consume(t[0]+.001)],
                       expected_reasons=['QUEUED','INVALID_INPUT','VISUAL_TARGET_HOLD']))
    return output


def execute_case(case):
    receiver=inbox(); events=[apply(receiver,event) for event in case['events']]
    return dict(name=case['name'],events=events,expected_reasons=case['expected_reasons'],
                passed=[r['answer']['reason'] for r in events]==case['expected_reasons'])


def source_hashes(parent):
    for name,expected in parent['sources'].items():
        if sha(ROOT/name)!=expected: raise ValueError('frozen source differs: '+name)
    return dict(parent['sources'],**{name:sha(ROOT/name) for name in OWN})


def page(report):
    live=report['live']; delivered=sum(r['consumption']['answer'].get('delivered',False) for r in live)
    rows=[]
    for r in live:
        age=r['consumption']['event']['now_s']-r['projection']['projection']['captured_at_s']
        rows.append(f'<tr><td>{r["offset_s"]} s</td><td>{r["load_started_at_s"]:.4f}</td>'
                    f'<td>{r["completed_at_s"]:.4f}</td><td>{age*1000:.1f} ms</td>'
                    f'<td>{r["receipt"]["answer"]["reason"]}</td><td>{r["consumption"]["answer"]["reason"]}</td></tr>')
    panels=[]
    for case in report['cases']:
        trace=' → '.join(r['answer']['reason'] for r in case['events'])
        panels.append(f'<details><summary>{case["name"]} · {"通过" if case["passed"] else "失败"} · {len(case["events"])} 个事件</summary>'
                      f'<p>{html.escape(trace)}</p><pre>{html.escape(json.dumps(case,ensure_ascii=False,indent=2))}</pre></details>')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实视觉观测流接收与消费</title><style>body{{font:16px/1.7 system-ui;background:#eef3f6;color:#183047;margin:0;padding:24px}}main{{max-width:1240px;margin:auto}}section{{background:white;padding:24px;border-radius:12px;margin:20px 0}}strong{{font-size:28px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dbe3e9;white-space:nowrap;text-align:left}}.scroll{{overflow:auto}}details{{padding:12px;border-bottom:1px solid #ddd}}summary{{cursor:pointer}}pre{{overflow:auto;max-height:400px;font-size:12px}}.warn{{color:#88451d}}a{{color:#125ea4}}</style>
<main><h1>从单张识别到有状态的观测消费</h1><section><strong>{delivered}/6</strong> 帧在实际消费时交付研究证据 · <strong>{sum(c['passed'] for c in report['cases'])}/10</strong> 组固定事件符合预期
<p>容量为 2 的先进先出队列。接收和消费分别检查时效；重复、乱序、错误坐标、错误时钟和队列溢出有明确结果。空检测不清空此前证据。</p>
<p class="warn">交付仅指同 TUM 世界系下的研究证据，不允许更新无人机导航地图或飞行。没有完成坐标对齐、可靠目标识别或避障控制。</p></section>
<section><h2>B：重新推理的六帧文件播放</h2><p>按原约两秒采集间隔，以单调墙钟等待、加载、推理、空间检查、接收和消费。处理时间实测；不是实时相机或高帧率稳定性验证。模型初始化 {report['startup_s']:.3f} 秒，在播放开始前完成。</p><div class="scroll"><table><thead><tr><th>图像</th><th>开始加载时刻</th><th>模型完成时刻</th><th>消费年龄</th><th>接收</th><th>消费</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>A：保存输出的十组事件重放</h2><p>使用前轮冻结的真实模型结果；额外延迟、暂停消费、错序和身份错误均为明确注入事件，未重新推理。展开查看逐次时间、结果与队列状态。</p>{''.join(panels)}</section>
<section><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定方案</a> · <a href="dataset-sources.json">数据来源</a></p><p>TUM RGB-D：J. Sturm 等，IROS 2012，CC BY 4.0。模型许可保留在 model-source。</p></section></main></html>'''


def run(dataset,baseline_dir,parent_dir,output,verify=False):
    from PIL import Image
    for directory in (dataset,baseline_dir,parent_dir): check_manifest(directory)
    verify_dataset(dataset)
    selection=json.loads((dataset/'selection.json').read_text(encoding='utf-8'))
    baseline=json.loads((baseline_dir/'report.json').read_text(encoding='utf-8'))
    parent=json.loads((parent_dir/'report.json').read_text(encoding='utf-8'))
    fixed=dict(sources=source_hashes(parent),baseline_sha256=sha(baseline_dir/'report.json'),
               parent_sha256=sha(parent_dir/'report.json'),dataset_sha256=sha(dataset/'manifest.json'))
    offline=[execute_case(case) for case in cases(parent)]
    if verify:
        count=check_manifest(output); report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if any(report[k]!=v for k,v in fixed.items()) or offline!=report['cases'] or not all(c['passed'] for c in offline):
            raise ValueError('fixed sources or case replay differs')
        if (output/'protocol.md').read_bytes()!=(ROOT/'docs/TIMED_INBOX_PROTOCOL.md').read_bytes(): raise ValueError('protocol differs')
        if len(report['live'])!=6 or report['session_exit_code']!=0: raise ValueError('live stream/session incomplete')
        if (output/'sensor/runtime-lock.json').read_bytes()!=(baseline_dir/'sensor/runtime-lock.json').read_bytes():
            raise ValueError('runtime differs')
        receiver=inbox(); points=0
        with EncodedVisionSession(output/'sensor',threads=16,input_mode='raw',replay=True) as detector:
            for index,row in enumerate(report['live']):
                chosen=selection['frames'][index]; observation=load_observation(dataset,chosen,selection['origin_s'])
                _,encoded=load_encoded(dataset,chosen,selection['origin_s'])
                record=row['call']; directory=output/'sensor'/record['directory']
                if row['index']!=index or record['index']!=index or row['offset_s']!=chosen['offset_s']:
                    raise ValueError('live order differs')
                ready=max(observation.captured_at_s,observation.depth_at_s,observation.pose_at_s)
                times=[ready,row['load_started_at_s'],row['loaded_at_s'],row['completed_at_s'],row['receipt']['event']['now_s'],row['consumption']['event']['now_s']]
                if times!=sorted(times): raise ValueError('live timestamp order differs')
                if json.loads((directory/'call.json').read_text(encoding='utf-8'))!=record or json.loads((directory/'result.json').read_text(encoding='utf-8'))!=record['result']:
                    raise ValueError('call/result differs')
                if (directory/'input.png').read_bytes()!=encoded.png_bytes: raise ValueError('original PNG differs')
                with Image.open(directory/'input.png') as image:
                    if prepare(image,record['window'])!=tensor_bytes(directory,'raw'): raise ValueError('input tensor differs')
                if detector.replay_frame(record)['boxes']!=record['result']['boxes']: raise ValueError('decode differs')
                eq=equivalence(output,baseline_dir,row,baseline['frames'][index])
                if eq!=row['equivalence'] or not all(eq.values()): raise ValueError('reference output differs')
                projected=project_record(observation,record,completed_at_s=row['completed_at_s'],now_s=row['completed_at_s'])
                if projected!=row['projection'] or row['receipt']['event']['result']!=projected: raise ValueError('projection differs')
                if apply(receiver,row['receipt']['event'])!=row['receipt'] or apply(receiver,row['consumption']['event'])!=row['consumption']:
                    raise ValueError('live receiver replay differs')
                answer=row['consumption']['answer']; age=row['consumption']['event']['now_s']-observation.captured_at_s
                if answer.get('delivered') and (age>.5+1e-9 or answer['world_frame']!=WORLD or answer['navigation_map_update_allowed']):
                    raise ValueError('late or unaligned navigation delivery')
                points+=independent_projection(observation,projected,chosen['pose'])
                print('verified live',chosen['offset_s'],flush=True)
        if page(report)!=(output/'demo.html').read_text(encoding='utf-8') or check_manifest(output)!=count:
            raise ValueError('page differs or replay changed archive')
        return dict(verified=True,live_calls=6,offline_cases=len(offline),offline_events=sum(len(c['events']) for c in offline),
                    files=count,sources=len(fixed['sources']),independent_samples=points,
                    delivered=sum(r['consumption']['answer'].get('delivered',False) for r in report['live']))
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(ROOT/'docs/TIMED_INBOX_PROTOCOL.md',output/'protocol.md')
    shutil.copyfile(dataset/'sources.json',output/'dataset-sources.json')
    (output/'model-source').mkdir()
    for name in ('LICENSE','NOTICE','sources.json'): shutil.copyfile(MODEL_ROOT/name,output/'model-source'/name)
    report=dict(**fixed,cases=offline,live=[])
    receiver=inbox()
    with EncodedVisionSession(output/'sensor',threads=16,input_mode='raw') as detector:
        report['startup_s']=detector.startup_s
        first=selection['frames'][0]
        first_ready=max(first[k][0]-selection['origin_s'] for k in ('rgb','depth','pose'))
        epoch=perf_counter()-first_ready
        for index,chosen in enumerate(selection['frames']):
            ready=max(chosen[k][0]-selection['origin_s'] for k in ('rgb','depth','pose'))
            remaining=ready-(perf_counter()-epoch)
            while remaining>0:
                sleep(min(remaining,.05)); remaining=ready-(perf_counter()-epoch)
            start=perf_counter()-epoch
            observation,encoded=load_encoded(dataset,chosen,selection['origin_s']); loaded=perf_counter()-epoch
            record=detector.detect_encoded(encoded); completed=perf_counter()-epoch
            projected=project_record(observation,record,completed_at_s=completed,now_s=completed)
            receipt=apply(receiver,dict(kind='submit',now_s=perf_counter()-epoch,result=projected))
            consumed=apply(receiver,dict(kind='consume',now_s=perf_counter()-epoch,target_world=WORLD))
            row=dict(index=index,offset_s=chosen['offset_s'],directory='sensor',input_mode='raw',threads=16,call=record,
                     load_started_at_s=start,loaded_at_s=loaded,completed_at_s=completed,projection=projected,
                     receipt=receipt,consumption=consumed)
            row['equivalence']=equivalence(output,baseline_dir,row,baseline['frames'][index]); report['live'].append(row)
            print('live',chosen['offset_s'],receipt['answer']['reason'],consumed['answer']['reason'],
                  round(consumed['event']['now_s']-observation.captured_at_s,4),'equivalent',all(row['equivalence'].values()),flush=True)
    report['session_exit_code']=detector.process.returncode
    if source_hashes(parent)!=fixed['sources']: raise ValueError('sources changed during playback')
    dump(output/'report.json',report); (output/'demo.html').write_text(page(report),encoding='utf-8')
    dump(output/'manifest.json',manifest(output))
    return dict(offline_cases=len(offline),offline_passed=sum(c['passed'] for c in offline),
                live_calls=6,delivered=sum(r['consumption']['answer'].get('delivered',False) for r in report['live']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=ROOT/'work/tum-rgbd-bag-input-01')
    parser.add_argument('--baseline',type=Path,default=ROOT/'work/tum-rgbd-01')
    parser.add_argument('--parent',type=Path,default=ROOT/'work/encoded-rgbd-01')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.baseline,args.parent,args.output,args.verify),indent=2),flush=True)
