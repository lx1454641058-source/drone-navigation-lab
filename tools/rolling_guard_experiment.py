"""来源：本项目原创。独立理想区间传感器与滚动运动学控制实验。"""
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.rolling_guard import RollingConfig,IntervalObservation,RollingGuard,WatchdogExecutor
from drone_nav.realvision import sha,dump
from tools.tum_rgbd_experiment import check_manifest,manifest

CAMERA='analytical-corridor-sensor';CLOCK='rolling-model-seconds'
OWN=('drone_nav/rolling_guard.py','tests/test_rolling_guard.py','tools/rolling_guard_experiment.py','docs/ROLLING_GUARD_PROTOCOL.md')
CASES=(('continuous','连续观测'),('dropout','一秒后断帧'),('late','观测延迟 0.46 秒'),
       ('wrong_clock','来源时钟错误'),('wrong_camera','来源身份错误'),('duplicate','重复帧'),
       ('missed_control','控制循环中断'),('wall','墙前停止'),('weak_brake','制动仅为声明的 30%'))


def sources(parent):
    for name,expected in parent['sources'].items():
        if sha(ROOT/name)!=expected: raise ValueError('frozen source differs: '+name)
    return dict(parent['sources'],**{name:sha(ROOT/name) for name in OWN})


def evaluate_trace(records,front_wall,config):
    x=v=0.;last_moving_at=0.;violations=[];max_error=0.
    for index,row in enumerate(records):
        before=row['execution']['before'];after=row['execution']['after'];start_time=before['now_s']
        if abs(before['x_m']-x)>1e-10 or abs(before['speed_mps']-v)>1e-10: raise ValueError('independent before-state differs')
        elapsed=0.;moving=False
        for part in row['execution']['parts']:
            dt=part['duration_s'];a=part['acceleration_mps2']
            active=min(dt,v/-a) if a<0 else dt
            if active>0 and (v>1e-12 or a>0):
                moving=True;last_moving_at=start_time+elapsed+active
            x+=v*active+.5*a*active*active;v=max(0.,v+a*active)
            if v<1e-12:v=0.
            elapsed+=dt
        max_error=max(max_error,abs(x-after['x_m']),abs(v-after['speed_mps']))
        if abs(elapsed-config.dt_s)>1e-10 or abs(after['now_s']-start_time-config.dt_s)>1e-9:
            raise ValueError('control tick duration differs')
        if moving:
            committed=row['execution']['committed']
            if committed is None: raise ValueError('uncommitted movement')
            b=committed['bound'];obs=committed['ticket']['observation']
            if before['x_m']-config.padding_m<b['lower_m']-1e-9 or x+config.padding_m>b['upper_m']+1e-9:
                violations.append(dict(step=index,kind='COMMITTED_ENVELOPE_EXCEEDED'))
            if last_moving_at>obs['captured_at_s']+config.ttl_s+1e-9:
                violations.append(dict(step=index,kind='MOVEMENT_AFTER_EVIDENCE_EXPIRY'))
        if x+config.body_radius_m>=front_wall-1e-9 or x-config.body_radius_m<=-1.+1e-9:
            violations.append(dict(step=index,kind='BODY_WALL_CONTACT'))
    if max_error>1e-10: raise ValueError('independent integration differs')
    return dict(steps=len(records),last_movement_at_s=last_moving_at,violations=violations,
                max_state_difference=max_error,final_x_m=x,final_speed_mps=v)


def scenario(name,label):
    config=RollingConfig();guard=RollingGuard(config,camera_id=CAMERA,clock_id=CLOCK)
    executor=WatchdogExecutor(config,camera_id=CAMERA,clock_id=CLOCK)
    front=.95 if name=='wall' else 3.;queue=[];events=[];records=[];status='TIME_BUDGET_HOLD'
    latency=.46 if name=='late' else .04;efficiency=.3 if name=='weak_brake' else 1.
    for tick in range(600):
        now=executor.now_s
        # World geometry is available only to this sensor and the independent evaluator.
        if tick%5==0 and not (name=='dropout' and now>=1.):
            frame=f'frame-{tick//5:04d}';camera=CAMERA;clock=CLOCK
            if now>=1.:
                if name=='wrong_camera':camera='wrong-source'
                if name=='wrong_clock':clock='wrong-clock'
                if name=='duplicate':frame='frame-0009'
            obs=IntervalObservation(frame,camera,clock,now,round(now+latency,10),
                max(-1.,executor.x_m-.8),min(front,executor.x_m+.8),'ideal full-cross-section static corridor sensor')
            queue.append(obs);events.append(dict(kind='capture',now_s=now,observation=asdict(obs)))
        delivered=[]
        while queue and queue[0].available_at_s<=now+1e-9:
            obs=queue.pop(0);answer=guard.receive(obs,now_s=now)
            event=dict(kind='delivery',now_s=now,frame_id=obs.frame_id,answer=answer)
            events.append(event);delivered.append(event)
        if name in ('missed_control','weak_brake') and now>=1.:
            ticket=None;decision=dict(reason='CONTROL_CALLBACK_MISSING',checks=[])
        else:
            ticket,decision=guard.choose(now_s=now,x_m=executor.x_m,speed_mps=executor.speed_mps,goal_m=1.)
        execution=executor.step(ticket,brake_efficiency=efficiency)
        records.append(dict(tick=tick,delivered=delivered,ticket_sent=ticket is not None,decision=decision,execution=execution))
        if executor.speed_mps==0.:
            if abs(executor.x_m-1.)<=.005:status='REACHED_AND_STOPPED';break
            if executor.x_m>1.005:status='GOAL_OVERSHOT';break
            if executor.mode=='STOPPED' and ticket is None:status='STOPPED_BEFORE_GOAL';break
            if guard.fault and executor.committed is None:status='FAULT_HOLD';break
    audit=evaluate_trace(records,front,config)
    if name=='continuous' and status!='REACHED_AND_STOPPED': raise ValueError('continuous case did not reach goal')
    if name=='late' and executor.x_m!=0.: raise ValueError('late observations caused movement')
    if name not in ('continuous','late') and (executor.speed_mps!=0. or executor.x_m>=.995):
        raise ValueError('fault/wall case did not stop before goal')
    if name!='weak_brake' and audit['violations']: raise ValueError('nominal model violated stopping commitment')
    if name=='weak_brake' and not audit['violations']: raise ValueError('brake-mismatch counterexample not exposed')
    return dict(name=name,label=label,config=asdict(config),front_wall_m=front,latency_s=latency,
        brake_efficiency=efficiency,status=status,events=events,records=records,audit=audit,
        final_time_s=executor.now_s,final_x_m=executor.x_m,final_speed_mps=executor.speed_mps,
        fault=guard.fault,model_only=True,flight_authorized=False)


def page(report):
    options=''.join(f'<option value="{i}">{r["label"]}</option>' for i,r in enumerate(report['cases']))
    panels=[];rows=[]
    for i,r in enumerate(report['cases']):
        points=' '.join(f'{60+800*s["execution"]["after"]["now_s"]/12:.2f},{290-220*s["execution"]["after"]["x_m"]:.2f}' for s in r['records'])
        speed=' '.join(f'{60+800*s["execution"]["after"]["now_s"]/12:.2f},{290-800*s["execution"]["after"]["speed_mps"]:.2f}' for s in r['records'])
        violations=len(r['audit']['violations']);delivery_count=sum(e['kind']=='delivery' for e in r['events'])
        note='声明的制动能力不成立，承诺已被突破；该反例保留为失败。' if violations else '本组在理想模型和输入契约下未发现承诺范围或期限突破。'
        panels.append(f'<section class="case" data-case="{i}" style="display:{"block" if i==0 else "none"}"><h2>{r["label"]}</h2><p>终态：{r["status"]} · 位置 {r["final_x_m"]:.4f} m · 速度 {r["final_speed_mps"]:.3f} m/s · 模型时间 {r["final_time_s"]:.2f} s</p><p>观测交付 {delivery_count} 次 · 控制步 {len(r["records"])} · 承诺违规记录 {violations}</p><svg viewBox="0 0 910 350" role="img" aria-label="位置和速度随模拟时间变化"><path d="M60 40V290H860" fill="none" stroke="#728292"/><path d="M60 70H860" stroke="#999" stroke-dasharray="4 4"/><polyline points="{points}" fill="none" stroke="#1773ae" stroke-width="2.5"/><polyline points="{speed}" fill="none" stroke="#c06520" stroke-width="2"/><text x="60" y="325">0 秒</text><text x="805" y="325">12 秒</text><text x="65" y="58">目标位置 1 m；蓝色为位置，橙色为速度（独立缩放）</text></svg><p>{note}</p><p>最后实际移动时刻：{r["audit"]["last_movement_at_s"]:.4f} 秒。静止等待不计为过期运动。</p><details><summary>查看独立核验与违规时刻</summary><pre>{html.escape(json.dumps(r["audit"],ensure_ascii=False,indent=2))}</pre></details></section>')
        rows.append(f'<tr><td>{r["label"]}</td><td>{r["status"]}</td><td>{r["final_x_m"]:.4f}</td><td>{len(r["records"])}</td><td>{violations}</td></tr>')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>滚动复查与观测中断停止</title><style>body{margin:0;padding:24px;background:#eef2f6;color:#193347;font:16px/1.7 system-ui}main{max-width:1100px;margin:auto}section{background:white;border-radius:12px;padding:22px;margin:20px 0}h1{font-size:30px}h2{font-size:22px}select{font:inherit;padding:9px;max-width:100%}svg{width:100%}pre{overflow:auto;max-height:350px;font-size:13px}table{border-collapse:collapse;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap}.scroll{overflow:auto}.warn{color:#914d20}a{color:#175b96}</style><main><h1>每次走一小段，同时保留停止余地</h1><section><p>控制周期 20 ms · 观测 10 Hz · 有效期 0.5 秒 · 不假设未来一定收到新帧</p><p class="warn">本页是一维运动学实验，使用理想完整走廊区间传感器，不是真实相机、AI 视觉或四旋翼动力学。所有结果均不授权实际飞行。</p><p>收到支持下一小段及其后停止范围的新观测，才发出该周期许可；缺少下一命令时执行上次保留的停止方案。回退开始后，迟到命令不能取消或延后停止。</p><label for="case">选择条件：</label><select id="case">'''+options+'''</select></section>'''+''.join(panels)+'''<section><h2>九组固定对照</h2><div class="scroll"><table><tr><th>条件</th><th>终态</th><th>最终位置 / m</th><th>模型控制步</th><th>承诺违规记录</th></tr>'''+''.join(rows)+'''</table></div><p>违规记录不是独立事故数量，同一步可能同时超过范围和期限。制动能力不足是故意保留的模型失效反例，不能算成功。</p></section><section><h2>复核资料</h2><p>记录每次采集/交付、许可依据、执行动作和状态；独立公式重算运动与停止时刻，并检查机体相对静态墙的位置。</p><p><a href="report.json">完整事件与轨迹</a> · <a href="protocol.md">固定方案和假设</a></p></section></main><script>document.getElementById('case').addEventListener('change',function(){document.querySelectorAll('.case').forEach(e=>e.style.display=e.dataset.case===this.value?'block':'none');});</script></html>'''


def run(parent_dir,output,verify=False):
    check_manifest(parent_dir);parent=json.loads((parent_dir/'report.json').read_text(encoding='utf-8'))
    fixed=sources(parent);report=dict(sources=fixed,parent_sha256=sha(parent_dir/'report.json'),
        input_scope='original analytic full-corridor sensor; no TUM frames consumed',model_only=True,new_model_calls=0,physical_flights=0,cases=[])
    for name,label in CASES:
        result=scenario(name,label);report['cases'].append(result)
        print(name,result['status'],round(result['final_x_m'],6),len(result['records']),len(result['audit']['violations']),flush=True)
    report['summary']=dict(cases=9,control_steps=sum(len(r['records']) for r in report['cases']),
        capture_events=sum(e['kind']=='capture' for r in report['cases'] for e in r['events']),
        delivery_events=sum(e['kind']=='delivery' for r in report['cases'] for e in r['events']),
        matched_model_cases_without_violations=sum(not r['audit']['violations'] for r in report['cases']),
        intentional_brake_mismatch_violations=len(report['cases'][-1]['audit']['violations']))
    if sources(parent)!=fixed: raise ValueError('source changed')
    if verify:
        count=check_manifest(output)
        if report!=json.loads((output/'report.json').read_text(encoding='utf-8')): raise ValueError('rolling event/state replay differs')
        if page(report)!=(output/'demo.html').read_text(encoding='utf-8'): raise ValueError('page replay differs')
        if (output/'protocol.md').read_bytes()!=(ROOT/'docs/ROLLING_GUARD_PROTOCOL.md').read_bytes(): raise ValueError('protocol differs')
        if check_manifest(output)!=count: raise ValueError('archive changed')
    else:
        output.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(ROOT/'docs/ROLLING_GUARD_PROTOCOL.md',output/'protocol.md')
        dump(output/'report.json',report);(output/'demo.html').write_text(page(report),encoding='utf-8')
        dump(output/'manifest.json',manifest(output));count=check_manifest(output)
    return dict(verified=verify,files=count,sources=len(fixed),**report['summary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,default=ROOT/'work/motion-volume-01')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();print(json.dumps(run(args.parent,args.output,args.verify),indent=2),flush=True)
