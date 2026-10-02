"""来源：本项目原创。保留父失败，验证真实移动提交的观测时效。"""
import argparse
from dataclasses import asdict,replace
import gzip
from html import escape
import json
from math import ceil,sqrt
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import textured_veto_experiment as prior
from tools import building_route_experiment as base
from drone_nav.actuator_vision import ActuatorVisionVehicle,VisualMotionLease
from drone_nav.textured_veto import PackedCameraDetector,finish_visual_veto
from drone_nav.semantic_sensor import packet_for,context_for,visual_move_decision
from drone_nav.vision_runtime import CPUVisionSession
from drone_nav.pinhole import Intrinsics,Pose
from drone_nav.textured_card import render_card
from tools.tinyformer_probe import prepare,MODEL,MODEL_SHA

PARENT=ROOT/'work/textured-veto-01'
OWN=('drone_nav/actuator_vision.py','tools/actuator_vision_experiment.py',
     'tests/test_actuator_vision.py','docs/ACTUATOR_VISION_PROTOCOL.md')


def specs():
    s=prior.specs()[0];s.update(commit_delay_s=0.)
    return [dict(s,**dict(dict(key=k,title=t),**changes)) for k,t,changes in (
        ('blank','灰色板：无更新时逐步撤销',{}),
        ('photo','照片板：正检出拒绝',dict(photo=True)),
        ('late-commit','提交前等待 0.6 秒',dict(commit_delay_s=.6)),
        ('changed-state','提交前等待 0.02 秒',dict(commit_delay_s=.02)),
        ('slow-perception','感知额外等待 0.6 秒',dict(semantic_delay_s=.6)),
        ('wrong-frame','照片检出：错误帧身份',dict(photo=True,semantic_wrong_frame=True)))]


class Vehicle(base.Vehicle,ActuatorVisionVehicle):
    capture=prior.Vehicle.capture

    def __init__(self,s,detector=None,saved=None,paired_raw=None):
        super().__init__(s,saved)
        self.detector=detector;self.texture=prior.texture_for(s['photo']);self.paired_raw=paired_raw
        self.semantic_checks=[];self.motor_checks=[];self.motion_attempts=[]
        self._motion_gate_active=False

    def observe_for_motion(self,target):
        before=self.state();index=len(self.semantic_checks)
        frame_index=self.frame_index if self.saved is not None else len(self.frames)
        frame=self.capture(Intrinsics(160,120,100,100,79.5,59.5),0,
            lambda xyz:Pose.look_at(xyz,(target[0]+.5,target[1]+.5,.8)))
        record=None
        if self.saved is not None:record=self.saved['semantic_checks'][index]
        elif self.paired_raw is not None:
            record=self.paired_raw['semantic_checks'][index]
            if (base.canonical(asdict(frame))!=self.paired_raw['frames'][record['frame_index']]['frame']
                or before!=record['actual_before']):raise ValueError('paired capture mismatch')
        call=record['call'] if record is not None else self.detector.detect(frame)
        wait_steps=ceil((call['elapsed_s']+self.spec['semantic_delay_s'])/self.physics.dt)
        self.hold(wait_steps*self.physics.dt)
        packet=packet_for(call,frame,frame_id=f'actuator-visual-{index}',captured_at_s=before['time_s'],completed_at_s=self.state()['time_s'])
        context=context_for(frame,packet.frame_id,before['time_s'])
        if self.spec['semantic_wrong_frame']:context=replace(context,frame_id='wrong-actuator-frame')
        started=perf_counter();candidate=visual_move_decision(packet,context,now_s=self.state()['time_s'])
        projection_s=record['projection_s'] if record is not None else perf_counter()-started
        projection_steps=ceil(projection_s/self.physics.dt);self.hold(projection_steps*self.physics.dt)
        decision=finish_visual_veto(candidate,packet,now_s=self.state()['time_s'])
        geometry=None;geometry_s=None;geometry_steps=0;lease=None;reason=decision['reason']
        if decision['permit_geometry_check']:
            started=perf_counter()
            geometry=self.sampling_history.check(self.confirmed_cell,target,now_s=self.state()['time_s'],
                budget=self.budget,assumption=self.sampling_assumption)
            geometry_s=record['geometry_s'] if record is not None else perf_counter()-started
            geometry_steps=ceil(geometry_s/self.physics.dt);self.hold(geometry_steps*self.physics.dt)
            decision=finish_visual_veto(candidate,packet,now_s=self.state()['time_s'])
            current=self.state();center=(self.confirmed_cell[0]+.5,self.confirmed_cell[1]+.5,3.5)
            stable=(sqrt(sum((a-b)**2 for a,b in zip(current['position'],center)))<=self.budget.position_tolerance_m
                    and sqrt(sum(v*v for v in current['velocity']))<=self.budget.speed_tolerance_mps)
            reason=decision['reason']
            if decision['permit_geometry_check'] and geometry['allowed'] and stable:
                h=self.sampling_history
                deadlines=[p['source']['valid_until_s'] for r in geometry['sampling'] for p in r['patches']]
                deadlines += [h.stamps[h.grid.free_seen[tuple(c)]]+self.budget.free_ttl_s for c in geometry['required_cells']]
                lease=VisualMotionLease(packet.frame_id,(target[0]+.5,target[1]+.5,3.5),packet.captured_at_s,
                                        current['time_s'],min(deadlines))
                reason='READY_FOR_ACTUATOR_CHECK'
            elif decision['permit_geometry_check']:
                reason='POST_VISION_GEOMETRY_HOLD' if not geometry['allowed'] else 'POST_VISION_POSE_HOLD'
        self.semantic_checks.append(dict(frame_index=frame_index,call=call,actual_before=before,actual_after=self.state(),
            target=list(target),packet=asdict(packet),context=asdict(context),waiting_steps=wait_steps,
            projection_s=projection_s,projection_waiting_steps=projection_steps,candidate=candidate,decision=decision,
            geometry=geometry,geometry_s=geometry_s,geometry_waiting_steps=geometry_steps,
            lease=None if lease is None else asdict(lease),preparation_reason=reason))
        return lease,reason


def simulate(s,model,detector=None,saved=None,paired_raw=None):
    v=Vehicle(s,detector,saved,paired_raw)
    try:
        result=base.MissionSupervisor(v,model,(3,8),tuple(s['goal']),max_ticks=s['max_ticks']).run()
        if v.history[-1]!=v.state():v.history.append(v.state())
        frames=v.frames if saved is None else saved['frames']
        raw=dict(frames=frames,commands=v.commands,contact_readings=v.contact_readings,
            semantic_checks=v.semantic_checks,motor_checks=v.motor_checks,motion_attempts=v.motion_attempts)
        summary=dict(spec=s,result=result,world=asdict(v.world),history=v.history,actual_final=v.state(),
            frames=len(frames),steps=v.steps,actions=v.actions,audit=v.audit,move_checks=v.move_checks,
            semantic_checks=v.semantic_checks,motor_checks=v.motor_checks,motion_attempts=v.motion_attempts,
            source_frames=v.source_frames,probes=v.probes,dll_sha256=v.physics.dll_sha256,model_sha256=v.physics.model_sha256)
        if saved is not None and (v.frame_index!=len(frames) or v.contact_index!=len(saved['contact_readings'])):
            raise ValueError('unused saved observations')
        return base.canonical(raw),base.canonical(summary)
    finally:v.close()


def audit_pair(raws):
    for key in ('late-commit','changed-state','slow-perception','wrong-frame'):
        source=raws['photo' if key=='wrong-frame' else 'blank'];other=raws[key]
        old=source['semantic_checks'][0];new=other['semantic_checks'][0]
        if len(other['semantic_checks'])!=1:raise ValueError('unexpected paired observation count')
        for field in ('call','actual_before','projection_s','target'):
            if old[field]!=new[field]:raise ValueError('paired field differs '+field)
        if source['frames'][:old['frame_index']+1]!=other['frames'][:new['frame_index']+1]:
            raise ValueError('paired capture prefix differs')
        if key in ('late-commit','changed-state'):
            for field in ('geometry','geometry_s','actual_after','lease'):
                if old[field]!=new[field]:raise ValueError('paired preparation differs '+field)
    return dict(exact_capture_prefix=True,exact_precommit_preparation=True,same_model_output_and_timings=True)


def page(report):
    rows=[];sections=[]
    for c in report['cases']:
        s=c['summary'];checks=s['motor_checks'];visual=s['semantic_checks'];r=s['result']
        values=[c['title'],[v['preparation_reason'] for v in visual],sum(x['allowed'] for x in checks),
                next((x['reason'] for x in checks if not x['allowed']),'无移动提交'),r['state']]
        rows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>')
        pictures=''.join(f'<img src="sensor/{escape(v["call"]["directory"])}/input.png" alt="模型实际相机输入">' for v in visual)
        detail=dict(visual=[dict(frame_id=v['packet']['frame_id'],captured_at_s=v['packet']['captured_at_s'],
            decision=v['decision'],lease=v['lease']) for v in visual],motor_checks=checks,
            motion_attempts=s['motion_attempts'],recovery=r['recovery']['state'] if r['recovery'] else None)
        sections.append('<section><h2>'+escape(c['title'])+'</h2>'+pictures+
            '<details><summary>移动推力检查与停止记录</summary><pre>'+escape(json.dumps(detail,ensure_ascii=False,indent=2))+'</pre></details></section>')
    return ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>移动执行处的视觉时效</title><style>body{background:#eef3f4;color:#193844;font:16px/1.7 system-ui;margin:0}main{max-width:1200px;margin:auto;padding:28px}.table{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}section{background:white;padding:20px;margin:20px 0}img{width:400px;max-width:100%;image-rendering:pixelated}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.notice{background:#fff0d8;padding:16px}summary{cursor:pointer}</style><main><h1>真正提交移动推力时，观测还有效吗？</h1><p>先完成几何扫描，再采集视觉；每个实际移动物理步都检查原有效期、指令状态与目标。过期后撤销继续移动，由原恢复控制器确认状态。</p><p class="notice">没有新观测就不续期，空检测也不证明没有障碍。本轮尚无运动中的连续感知，因此可能只开始短暂运动再中止；恢复推力仍会执行，不等于立即静止或现实空间安全。照片是二维道具，非真实三维行人。</p><div class="table"><table><tr><th>条件</th><th>移动准备</th><th>允许的移动步</th><th>首个提交拒绝</th><th>任务终态</th></tr>'''+''.join(rows)+'</table></div>'+''.join(sections)+
        '<p>照片来源 TUM RGB-D benchmark，J. Sturm 等，CC BY 4.0，本项目裁剪贴图。<a href="dataset-sources.json">来源</a> · <a href="report.json">完整报告</a> · <a href="protocol.md">固定协议</a></p></main></html>').encode()


def run(output,verify=False):
    base.check_manifest(PARENT);parent=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    sources={n:base.sha(ROOT/n) for n in sorted(set(parent['sources'])|set(OWN))}
    if any(sources[n]!=d for n,d in parent['sources'].items()):raise ValueError('frozen parent changed')
    if base.sha(MODEL)!=MODEL_SHA:raise ValueError('model changed')
    model=base.ColorModel.from_dict(json.loads(base.MODEL.read_text(encoding='utf-8')))
    if verify:
        from PIL import Image
        files=base.check_manifest(output);report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if report['sources']!=sources or report['parent_sha256']!=base.sha(PARENT/'report.json'):raise ValueError('source binding changed')
        if [c['summary']['spec'] for c in report['cases']]!=specs():raise ValueError('matrix changed')
        raws={};calls=set()
        with CPUVisionSession(output/'sensor',threads=16,input_mode='raw',replay=True) as detector:
            for c in report['cases']:
                raw=json.loads(gzip.decompress((output/c['input']).read_bytes()));raws[c['key']]=raw
                for v in raw['semantic_checks']:
                    call=v['call'];folder=output/'sensor'/call['directory'];f=raw['frames'][v['frame_index']]['frame']
                    if call['directory'] not in calls:detector.replay_frame(call);calls.add(call['directory'])
                    with Image.open(folder/'input.png') as im:
                        if im.tobytes()!=bytes(v for rgb in f['rgb'] for v in rgb):raise ValueError('RGB mismatch')
                        if prepare(im,call['window'])!=(folder/'input.f32').read_bytes():raise ValueError('tensor mismatch')
                    if call['result']!=json.loads((folder/'result.json').read_text(encoding='utf-8')):raise ValueError('detector output mismatch')
                new,s=simulate(c['summary']['spec'],model,saved=raw)
                if new!=raw or s!=c['summary']:raise ValueError('control replay mismatch '+c['key'])
                world=base.saved_world(s['world'])
                with base.QuadrotorPhysics(model_xml=base.world_xml(world)) as physics:
                    if base.replay_raw(physics,raw,(3,8))!=s['history'] or physics.state()!=s['actual_final']:raise ValueError('motor replay mismatch')
                texture=prior.texture_for(s['spec']['photo'])
                for item in raw['frames']:
                    d=item['frame'];pose=Pose(**{k:tuple(v) for k,v in d['pose'].items()})
                    frame,paint=render_card(world,pose,Intrinsics(**d['intrinsics']),card_name='card',texture=texture,tick=d['tick'],max_range_m=s['spec']['range_m'])
                    if base.canonical(asdict(frame))!=d or base.canonical(paint)!=item['paint']:raise ValueError('rerender mismatch')
                print('verified',c['key'],flush=True)
        if audit_pair(raws)!=report['paired'] or page(report)!=(output/'demo.html').read_bytes():raise ValueError('paired/page mismatch')
        return dict(verified=True,files=files,sources=len(sources),cases=len(raws),frames=sum(len(r['frames']) for r in raws.values()),steps=sum(len(r['commands']) for r in raws.values()))
    if output.exists():raise FileExistsError('refuse overwrite')
    output.mkdir(parents=True);raws={};cases=[]
    with CPUVisionSession(output/'sensor',threads=16,input_mode='raw') as session:
        detector=PackedCameraDetector(session)
        for s in specs():
            paired=None if s['key'] in ('blank','photo') else raws['photo' if s['key']=='wrong-frame' else 'blank']
            raw,summary=simulate(s,model,detector,paired_raw=paired);raws[s['key']]=raw
            filename=s['key']+'/raw.json.gz'
            base.put(output/filename,gzip.compress(json.dumps(raw,separators=(',',':')).encode(),mtime=0))
            base.dump(output/s['key']/'summary.json',summary)
            cases.append(dict(key=s['key'],title=s['title'],input=filename,summary=summary))
            print(s['key'],summary['result']['state'],summary['result']['original_reason'],len(summary['motor_checks']),flush=True)
        startup=session.startup_s;new_calls=len(session.records)
    if sources!={n:base.sha(ROOT/n) for n in sources}:raise ValueError('source changed during run')
    report=dict(kind='actuator-vision',sources=sources,parent_sha256=base.sha(PARENT/'report.json'),
        new_model_calls=new_calls,model_startup_s=startup,cases=cases,paired=audit_pair(raws))
    base.dump(output/'report.json',report);base.put(output/'demo.html',page(report))
    base.put(output/'protocol.md',(ROOT/'docs/ACTUATOR_VISION_PROTOCOL.md').read_bytes())
    for name in ('dataset-sources.json','texture-source.png','model-source/LICENSE','model-source/NOTICE','model-source/sources.json'):
        base.put(output/name,(PARENT/name).read_bytes())
    base.dump(output/'manifest.json',{p.relative_to(output).as_posix():base.sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases),frames=sum(c['summary']['frames'] for c in cases),steps=sum(c['summary']['steps'] for c in cases),new_model_calls=new_calls)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.verify),ensure_ascii=False))
