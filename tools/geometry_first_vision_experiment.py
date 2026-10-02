"""来源：本项目原创。先计算几何，再获取新视觉；复用原实际提交检查。"""
import argparse
from dataclasses import asdict,replace
import gzip
import json
from math import ceil,sqrt
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import actuator_vision_experiment as old
from tools import building_route_experiment as base
from tools import textured_veto_experiment as textured
from drone_nav.actuator_vision import VisualMotionLease
from drone_nav.textured_veto import PackedCameraDetector,finish_visual_veto
from drone_nav.semantic_sensor import packet_for,context_for,visual_move_decision
from drone_nav.vision_runtime import CPUVisionSession
from drone_nav.pinhole import Intrinsics,Pose
from drone_nav.textured_card import render_card
from tools.tinyformer_probe import MODEL,MODEL_SHA,prepare

PARENT=ROOT/'work/actuator-vision-01'
OWN=('tools/geometry_first_vision_experiment.py','docs/GEOMETRY_FIRST_VISION_PROTOCOL.md')
specs=old.specs
page=old.page
audit_pair=old.audit_pair


class Vehicle(old.Vehicle):
    def observe_for_motion(self,target):
        index=len(self.semantic_checks);record=None
        if self.saved is not None:record=self.saved['semantic_checks'][index]
        elif self.paired_raw is not None:record=self.paired_raw['semantic_checks'][index]
        geometry_before=self.state();started=perf_counter()
        geometry=self.sampling_history.check(self.confirmed_cell,target,now_s=geometry_before['time_s'],
            budget=self.budget,assumption=self.sampling_assumption)
        geometry_s=record['geometry_s'] if record is not None else perf_counter()-started
        geometry_steps=ceil(geometry_s/self.physics.dt);self.hold(geometry_steps*self.physics.dt)
        before=self.state();frame_index=self.frame_index if self.saved is not None else len(self.frames)
        frame=self.capture(Intrinsics(160,120,100,100,79.5,59.5),0,
            lambda xyz:Pose.look_at(xyz,(target[0]+.5,target[1]+.5,.8)))
        if self.saved is None and record is not None:
            if (base.canonical(asdict(frame))!=self.paired_raw['frames'][record['frame_index']]['frame']
                or before!=record['actual_before'] or geometry_before!=record['geometry_before']
                or base.canonical(geometry)!=record['geometry']):raise ValueError('paired geometry/capture mismatch')
        call=record['call'] if record is not None else self.detector.detect(frame)
        wait_steps=ceil((call['elapsed_s']+self.spec['semantic_delay_s'])/self.physics.dt)
        self.hold(wait_steps*self.physics.dt)
        packet=packet_for(call,frame,frame_id=f'actuator-visual-{index}',captured_at_s=before['time_s'],completed_at_s=self.state()['time_s'])
        context=context_for(frame,packet.frame_id,before['time_s'])
        if self.spec['semantic_wrong_frame']:context=replace(context,frame_id='wrong-actuator-frame')
        started=perf_counter();candidate=visual_move_decision(packet,context,now_s=self.state()['time_s'])
        projection_s=record['projection_s'] if record is not None else perf_counter()-started
        projection_steps=ceil(projection_s/self.physics.dt);self.hold(projection_steps*self.physics.dt)
        current=self.state();decision=finish_visual_veto(candidate,packet,now_s=current['time_s'])
        center=(self.confirmed_cell[0]+.5,self.confirmed_cell[1]+.5,3.5)
        stable=(sqrt(sum((a-b)**2 for a,b in zip(current['position'],center)))<=self.budget.position_tolerance_m
                and sqrt(sum(v*v for v in current['velocity']))<=self.budget.speed_tolerance_mps)
        lease=None;reason=decision['reason']
        if decision['permit_geometry_check'] and geometry['allowed'] and stable:
            h=self.sampling_history
            deadlines=[p['source']['valid_until_s'] for r in geometry['sampling'] for p in r['patches']]
            deadlines += [h.stamps[h.grid.free_seen[tuple(c)]]+self.budget.free_ttl_s for c in geometry['required_cells']]
            lease=VisualMotionLease(packet.frame_id,(target[0]+.5,target[1]+.5,3.5),packet.captured_at_s,
                                    current['time_s'],min(deadlines))
            reason='READY_FOR_ACTUATOR_CHECK'
        elif decision['permit_geometry_check']:
            reason='POST_VISION_GEOMETRY_HOLD' if not geometry['allowed'] else 'POST_VISION_POSE_HOLD'
        self.semantic_checks.append(dict(frame_index=frame_index,call=call,actual_before=before,actual_after=current,
            target=list(target),packet=asdict(packet),context=asdict(context),waiting_steps=wait_steps,
            projection_s=projection_s,projection_waiting_steps=projection_steps,candidate=candidate,decision=decision,
            geometry=geometry,geometry_s=geometry_s,geometry_waiting_steps=geometry_steps,geometry_before=geometry_before,
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
                texture=textured.texture_for(s['spec']['photo'])
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
    report=dict(kind='geometry-first-actuator-vision',sources=sources,parent_sha256=base.sha(PARENT/'report.json'),
        new_model_calls=new_calls,model_startup_s=startup,cases=cases,paired=audit_pair(raws))
    base.dump(output/'report.json',report);base.put(output/'demo.html',page(report))
    base.put(output/'protocol.md',(ROOT/'docs/GEOMETRY_FIRST_VISION_PROTOCOL.md').read_bytes())
    for name in ('dataset-sources.json','texture-source.png','model-source/LICENSE','model-source/NOTICE','model-source/sources.json'):
        base.put(output/name,(PARENT/name).read_bytes())
    base.dump(output/'manifest.json',{p.relative_to(output).as_posix():base.sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases),frames=sum(c['summary']['frames'] for c in cases),steps=sum(c['summary']['steps'] for c in cases),new_model_calls=new_calls)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.verify),ensure_ascii=False))
