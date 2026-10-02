"""来源：本项目原创。照片道具、真实模型与物理导航的固定开发对照。"""
import argparse
from dataclasses import asdict
import gzip
from html import escape
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import building_route_experiment as base
from drone_nav.textured_card import CardTexture,render_card
from drone_nav.textured_veto import TexturedVetoMixin,PackedCameraDetector
from drone_nav.physical_vehicle import rotate_vector
from drone_nav.verify_exploration import RecordedCamera
from drone_nav.range_observation import RangeFrame
from drone_nav.pinhole import Pose,Intrinsics
from drone_nav.vision_runtime import CPUVisionSession
from tools.tinyformer_probe import MODEL,MODEL_SHA,prepare

PARENT=ROOT/'work/semantic-move-01'
PHOTO=ROOT/'work/tum-rgbd-bag-input-01/rgb/1341846315.566107512.png'
PHOTO_SHA='f898cde1e558b559af70a0033c388a5f930cee0627199c5be7d3616abe04c974'
CROP=(344,35,593,480)
OWN=('drone_nav/textured_card.py','drone_nav/textured_veto.py','drone_nav/vision_runtime.py',
     'drone_nav/packed_rgbd.py','tools/vision_runtime_worker.cjs','tools/textured_veto_experiment.py',
     'tests/test_textured_card.py','tests/test_textured_veto.py','docs/TEXTURED_VETO_PROTOCOL.md')


def specs():
    s=base.base_spec()
    s.update(goal=[4,8],max_ticks=4,wide_scan=True,level_scan=True,mission_goal_probe=True,
        buildings=[dict(name='card',low=[6.,8.,0.],high=[6.2,9.,2.2])],
        semantic_delay_s=0.,semantic_wrong_frame=False,semantic_veto_enabled=True,photo=True)
    return [dict(s,**dict(dict(key=key,title=title),**changes)) for key,title,changes in (
        ('blank','灰色板：实际模型',dict(photo=False)),
        ('photo','人物照片板：实际模型',{}),
        ('ablation','相同检测与计时：仅关闭目标否决',dict(semantic_veto_enabled=False)),
        ('delayed','相同检测：额外等待 0.6 秒',dict(semantic_delay_s=.6)),
        ('wrong-frame','相同检测：错误帧身份',dict(semantic_wrong_frame=True)))]


def texture_for(photo):
    from PIL import Image
    if not photo:return CardTexture(1,1,bytes((145,145,145)))
    if base.sha(PHOTO)!=PHOTO_SHA:raise ValueError('source photo changed')
    with Image.open(PHOTO) as image:
        crop=image.convert('RGB').crop(CROP)
        return CardTexture(crop.width,crop.height,crop.tobytes())


class Vehicle(TexturedVetoMixin,base.Vehicle):
    def __init__(self,s,detector=None,saved=None,paired_raw=None):
        super().__init__(s,saved)
        self.detector=detector;self.paired_raw=paired_raw;self.semantic_checks=[]
        self.texture=texture_for(s['photo'])

    def capture(self,k,tick,aim=None):
        state=self.state();xyz=tuple(state['position']);q=state['quaternion']
        camera=Pose.look_at(xyz,(xyz[0],xyz[1],0)) if aim is None else aim(xyz)
        pose=Pose(xyz,rotate_vector(camera.right,q),rotate_vector(camera.down,q),rotate_vector(camera.forward,q))
        if self.saved is None:
            frame,paint=render_card(self.world,pose,k,card_name='card',texture=self.texture,
                                   tick=tick,max_range_m=self.spec['range_m'])
            self.frames.append(dict(frame=asdict(frame),capture_state=state,sensor_fault_active=False,paint=paint))
        else:
            item=self.saved_frames[self.frame_index]
            if state!=item['capture_state'] or item['sensor_fault_active']:
                raise ValueError('textured capture state/fault mismatch')
            old=RecordedCamera([item['frame']]).capture(pose,k,tick);data=item['frame']
            frame=RangeFrame(old.intrinsics,old.pose,old.rgb,old.depth_z_m,old.tick,
                            tuple(data['outcomes']),data['max_range_m'],data['sensor_model'])
            frame.validate();self.frame_index+=1
        if self._scan_records is not None:self._scan_records.append((state,frame))
        return frame


def simulate(s,model,detector=None,saved=None,paired_raw=None):
    vehicle=Vehicle(s,detector,saved,paired_raw)
    try:
        result=base.MissionSupervisor(vehicle,model,(3,8),tuple(s['goal']),max_ticks=s['max_ticks']).run()
        if vehicle.history[-1]!=vehicle.state():vehicle.history.append(vehicle.state())
        frames=vehicle.frames if saved is None else saved['frames']
        raw=dict(frames=frames,commands=vehicle.commands,contact_readings=vehicle.contact_readings,
                 semantic_checks=vehicle.semantic_checks)
        h=vehicle.sampling_history
        summary=dict(spec=s,result=result,world=asdict(vehicle.world),history=vehicle.history,
            actual_final=vehicle.state(),frames=len(frames),steps=vehicle.steps,actions=vehicle.actions,
            audit=vehicle.audit,move_checks=vehicle.move_checks,semantic_checks=vehicle.semantic_checks,
            source_frames=vehicle.source_frames,probes=vehicle.probes,
            cache=dict(peak_frames=h.peak_frames,retired=h.retired),
            dll_sha256=vehicle.physics.dll_sha256,model_sha256=vehicle.physics.model_sha256)
        if saved is not None and (vehicle.frame_index!=len(frames)
            or vehicle.contact_index!=len(saved['contact_readings'])
            or len(vehicle.semantic_checks)!=len(saved['semantic_checks'])):
            raise ValueError('unused saved observations')
        return base.canonical(raw),base.canonical(summary)
    finally:vehicle.close()


def paired_check(cases,raws):
    original=raws['photo'];checks=original['semantic_checks']
    if len(checks)!=1:raise ValueError('expected one photo observation in this fixed matrix')
    first=checks[0]
    for key in ('ablation','delayed','wrong-frame'):
        raw=raws[key]
        if len(raw['semantic_checks'])!=1:raise ValueError('paired case changed observation count')
        new=raw['semantic_checks'][0]
        for field in ('call','actual_before','projection_s','target'):
            if new[field]!=first[field]:raise ValueError('paired mismatch '+key+' '+field)
        end=new['frame_index']+1;old_end=first['frame_index']+1
        if raw['frames'][:end]!=original['frames'][:old_end]:raise ValueError('paired capture prefix differs')
        if key=='ablation' and new['actual_after']!=first['actual_after']:
            raise ValueError('paired wait state differs')
    return dict(exact_capture_prefix=True,same_model_output=True,same_measured_latency=True,
                ablation_same_state_before_geometry=True)


def page(report):
    rows=[];cards=[]
    for case in report['cases']:
        s=case['summary'];checks=s['semantic_checks'];r=s['result']
        values=[case['title'],r['state'],r['confirmed_cell'],len(s['actions']),
                [c['decision']['reason'] for c in checks],r['original_reason']]
        rows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>')
        pictures=[]
        for c in checks:
            call=c['call'];label=f"原始框 {len(call['result']['boxes'])}；模型流程 {call['elapsed_s']:.3f} 秒；判断年龄 {c['decision']['age_s']:.3f} 秒"
            pictures.append(f'<figure><img src="sensor/{escape(call["directory"])}/input.png" alt="模型实际输入"><figcaption>{escape(label)}</figcaption></figure>')
        cards.append('<section><h2>'+escape(case['title'])+'</h2>'+''.join(pictures)+
            '<details><summary>判定、动作和任务回执</summary><pre>'+escape(json.dumps(dict(checks=checks,actions=s['actions'],result=r),ensure_ascii=False,indent=2))+'</pre></details></section>')
    return ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>照片目标与导航停止对照</title>
<style>body{margin:0;background:#eef3f4;color:#193844;font:16px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:28px}p{line-height:1.8}.notice{background:#fff1d8;padding:18px}.table{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}section{background:white;padding:20px;margin:20px 0;border-radius:8px}img{width:480px;max-width:100%;image-rendering:pixelated}figure{margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer}a{color:#176277}</style>
<main><h1>实际检出目标后，物理任务会怎样？</h1><p>模型读取同帧仿真 RGB，照片外观与射线深度来自同一个板面。后面三组复用照片组的实际输出与计时，比较目标否决、延迟和帧身份。</p><p class="notice">照片板位于目标后方，检测到它即停止属于保守策略，不证明避免了一次碰撞。二维道具不是三维行人，原始框有重复；本轮仍非真实相机持续实时导航。查看保存结果不会重新推理。</p>
<div class="table"><table><tr><th>条件</th><th>任务终态</th><th>确认格</th><th>移动动作数</th><th>视觉决定</th><th>任务原因</th></tr>'''+''.join(rows)+'</table></div>'+''.join(cards)+
        '<p>照片：TUM RGB-D benchmark，J. Sturm 等，CC BY 4.0；本项目裁剪并贴图。<a href="dataset-sources.json">来源与修改</a> · <a href="report.json">完整报告</a> · <a href="protocol.md">协议</a> · <a href="manifest.json">文件摘要</a></p></main></html>').encode('utf-8')


def run(output,verify=False):
    base.check_manifest(PARENT)
    parent=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    sources={name:base.sha(ROOT/name) for name in sorted(set(parent['sources'])|set(OWN))}
    if any(sources[n]!=d for n,d in parent['sources'].items()):raise ValueError('parent source changed')
    if base.sha(MODEL)!=MODEL_SHA or base.sha(PHOTO)!=PHOTO_SHA:raise ValueError('fixed input changed')
    base.check_manifest(base.MODEL.parent)
    model=base.ColorModel.from_dict(json.loads(base.MODEL.read_text(encoding='utf-8')))
    if verify:
        from PIL import Image
        files=base.check_manifest(output)
        report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if (report['sources']!=sources or report['parent_sha256']!=base.sha(PARENT/'report.json')
            or report['photo_sha256']!=PHOTO_SHA or report['model_sha256']!=MODEL_SHA):
            raise ValueError('archive sources changed')
        if [c['summary']['spec'] for c in report['cases']]!=specs():raise ValueError('matrix changed')
        raws={};calls=set();frames=steps=0
        with CPUVisionSession(output/'sensor',threads=16,input_mode='raw',replay=True) as detector:
            for case in report['cases']:
                raw=json.loads(gzip.decompress((output/case['input']).read_bytes()));raws[case['key']]=raw
                for check in raw['semantic_checks']:
                    call=check['call'];folder=output/'sensor'/call['directory']
                    if call['directory'] not in calls:detector.replay_frame(call);calls.add(call['directory'])
                    frame=raw['frames'][check['frame_index']]['frame']
                    with Image.open(folder/'input.png') as im:
                        if list(im.getdata())!=[tuple(p) for p in frame['rgb']]:raise ValueError('RGB input mismatch')
                        if prepare(im,call['window'])!=(folder/'input.f32').read_bytes():raise ValueError('tensor input mismatch')
                    if base.sha(folder/'input.png')!=call['input_rgb_sha256']:raise ValueError('PNG changed')
                    inner=json.loads((folder/'call.json').read_text(encoding='utf-8'))
                    expected=dict(call);expected['elapsed_s']=expected.pop('inner_elapsed_s')
                    if expected!=inner:raise ValueError('model call binding changed')
                new,summary=simulate(case['summary']['spec'],model,saved=raw)
                if new!=raw or summary!=case['summary']:raise ValueError('control replay differs '+case['key'])
                world=base.saved_world(summary['world'])
                with base.QuadrotorPhysics(model_xml=base.world_xml(world)) as physics:
                    if base.replay_raw(physics,raw,(3,8))!=summary['history'] or physics.state()!=summary['actual_final']:
                        raise ValueError('independent motor replay differs')
                texture=texture_for(summary['spec']['photo'])
                for item in raw['frames']:
                    d=item['frame'];pose=Pose(**{k:tuple(v) for k,v in d['pose'].items()})
                    rendered,paint=render_card(world,pose,Intrinsics(**d['intrinsics']),card_name='card',texture=texture,
                        tick=d['tick'],max_range_m=summary['spec']['range_m'])
                    if base.canonical(asdict(rendered))!=d or base.canonical(paint)!=item['paint']:
                        raise ValueError('textured rerender differs')
                frames+=summary['frames'];steps+=summary['steps']
                print('verified',case['key'],flush=True)
        if paired_check(report['cases'],raws)!=report['paired']:raise ValueError('paired audit changed')
        if page(report)!=(output/'demo.html').read_bytes():raise ValueError('page differs')
        return dict(verified=True,cases=len(raws),files=files,sources=len(sources),frames=frames,steps=steps,model_calls_replayed=len(calls))
    if output.exists():raise FileExistsError('refuse overwrite '+str(output))
    output.mkdir(parents=True)
    cases=[];raws={}
    with CPUVisionSession(output/'sensor',threads=16,input_mode='raw') as session:
        detector=PackedCameraDetector(session)
        for spec in specs():
            raw,summary=simulate(spec,model,detector,paired_raw=raws.get('photo') if spec['key'] not in ('blank','photo') else None)
            filename=spec['key']+'/raw.json.gz'
            base.put(output/filename,gzip.compress(json.dumps(raw,separators=(',',':')).encode(),mtime=0))
            case=dict(key=spec['key'],title=spec['title'],input=filename,summary=summary)
            base.dump(output/spec['key']/'summary.json',summary)
            cases.append(case);raws[spec['key']]=raw
            print(spec['key'],summary['result']['state'],summary['result']['original_reason'],
                  [c['decision']['reason'] for c in summary['semantic_checks']],flush=True)
        startup=session.startup_s;new_calls=len(session.records)
    if sources!={n:base.sha(ROOT/n) for n in sources}:raise ValueError('source changed during run')
    report=dict(kind='textured-visual-veto',sources=sources,parent_sha256=base.sha(PARENT/'report.json'),
        photo_sha256=PHOTO_SHA,crop=list(CROP),model_sha256=MODEL_SHA,model_startup_s=startup,new_model_calls=new_calls,
        paired=paired_check(cases,raws),cases=cases)
    attribution=json.loads((ROOT/'work/tum-rgbd-bag-input-01/sources.json').read_text(encoding='utf-8'))
    attribution.update(new_modifications='Crop (344,35,593,480), nearest-neighbor perspective texture on low-X box face; no source depth or pose used',source_photo_sha256=PHOTO_SHA)
    base.dump(output/'dataset-sources.json',attribution)
    base.dump(output/'report.json',report)
    base.put(output/'protocol.md',(ROOT/'docs/TEXTURED_VETO_PROTOCOL.md').read_bytes())
    base.put(output/'demo.html',page(report))
    for name in ('LICENSE','NOTICE','sources.json'):
        base.put(output/'model-source'/name,(MODEL.parent/name).read_bytes())
    base.put(output/'texture-source.png',PHOTO.read_bytes())
    base.dump(output/'manifest.json',{p.relative_to(output).as_posix():base.sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases),frames=sum(c['summary']['frames'] for c in cases),
        steps=sum(c['summary']['steps'] for c in cases),new_model_calls=new_calls)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();print(json.dumps(run(args.output,args.verify),ensure_ascii=False))
