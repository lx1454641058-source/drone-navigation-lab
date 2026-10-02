"""来源：本项目原创。真实神经网络输出进入原物理移动前检查。"""
import argparse
from dataclasses import asdict, replace
import gzip
from html import escape
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import building_route_experiment as base
from drone_nav.semantic_move import SemanticMoveMixin
from drone_nav.semantic_sensor import TinyFormerSession
from drone_nav.raycast import World,Box
from tools.tinyformer_probe import prepare,MODEL,MODEL_SHA
from drone_nav.pinhole import Pose,Intrinsics
from drone_nav.range_observation import render_range

PARENT=ROOT/'work/simulation-building-01'
OWN=('drone_nav/semantic_sensor.py','drone_nav/semantic_move.py','tools/semantic_sensor_worker.cjs',
     'tools/semantic_move_experiment.py','tests/test_semantic_move.py','docs/SEMANTIC_MOVE_PROTOCOL.md',
     'drone_nav/detection_bridge.py','drone_nav/tinyformer_math.cjs','tools/tinyformer_probe.py')


def specs():
    s=base.base_spec()
    s.update(goal=[4,8],max_ticks=4,buildings=[],wide_scan=True,level_scan=True,mission_goal_probe=True,
             semantic_delay_s=0.,semantic_wrong_frame=False,semantic_worker_error=False)
    rows=[]
    for key,title,changes in (
        ('clear','同帧神经网络检查',{}),
        ('delayed','额外延迟 0.6 秒',dict(semantic_delay_s=.6)),
        ('wrong-frame','深度来自另一帧',dict(semantic_wrong_frame=True)),
        ('person','简化行人：保留模型漏检',dict(buildings=[dict(name='person',low=[4.2,8.2,0.],high=[4.8,8.8,1.7])])),
        ('worker-error','检测器响应异常',dict(semantic_worker_error=True))):
        rows.append(dict(s,**dict(dict(key=key,title=title),**changes)))
    return rows


class FailedDetector:
    def detect(self,frame):raise ValueError('injected detector response failure')


class Vehicle(SemanticMoveMixin,base.Vehicle):
    def __init__(self,s,detector,saved=None):
        super().__init__(s,saved)
        self.detector=FailedDetector() if s['semantic_worker_error'] else detector
        self.semantic_checks=[]
        # Only the rendered material changes to the pre-existing person class.
        # Physical geometry stays byte-identical to the world allocated above.
        self.world=World(tuple(replace(b,label=4) if isinstance(b,Box) and b.name=='person' else b
                               for b in self.world.surfaces))
        if base.world_xml(self.world)!=base.world_xml(base.world_for(s)):
            raise ValueError('semantic material unexpectedly changed physics geometry')


def simulate(s,model,detector=None,saved=None):
    v=Vehicle(s,detector,saved)
    try:
        result=base.MissionSupervisor(v,model,(3,8),tuple(s['goal']),max_ticks=s['max_ticks']).run()
        if v.history[-1]!=v.state():v.history.append(v.state())
        frames=v.frames if saved is None else saved['frames']
        raw=dict(frames=frames,commands=v.commands,contact_readings=v.contact_readings,semantic_checks=v.semantic_checks)
        h=v.sampling_history
        summary=dict(spec=s,result=result,world=asdict(v.world),history=v.history,actual_final=v.state(),
            frames=len(frames),steps=v.steps,actions=v.actions,audit=v.audit,move_checks=v.move_checks,
            semantic_checks=v.semantic_checks,source_frames=v.source_frames,probes=v.probes,
            cache=dict(peak_frames=h.peak_frames,retired=h.retired),
            dll_sha256=v.physics.dll_sha256,model_sha256=v.physics.model_sha256)
        if saved is not None and (v.frame_index!=len(frames) or v.contact_index!=len(saved['contact_readings'])
                                  or len(v.semantic_checks)!=len(saved['semantic_checks'])):
            raise ValueError('unused saved observations')
        return base.canonical(raw),base.canonical(summary)
    finally:v.close()


def page(report):
    rows=[];cards=[]
    for case in report['cases']:
        s=case['summary'];r=s['result'];checks=s['semantic_checks']
        values=[case['title'],r['state'],r['confirmed_cell'],
                [c['decision']['reason'] for c in checks],r['original_reason'] or '接地确认']
        rows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>')
        pictures=[]
        for c in checks:
            call=c['call']
            if 'error' in call:continue
            label=f"检测框 {len(call['result']['boxes'])}；处理 {call['elapsed_s']:.3f} 秒；物理等待 {c['waiting_steps']*.002:.3f} 秒"
            pictures.append(f'<figure><img src="sensor/{escape(call["directory"])}/input.png" alt="模型实际读取的仿真相机画面"><figcaption>{escape(label)}</figcaption></figure>')
        detail=dict(semantic_checks=checks,actions=s['actions'],audit=s['audit'])
        cards.append('<section><h2>'+escape(case['title'])+'</h2>'+''.join(pictures)+
            '<details><summary>查看时间、来源、判定与动作回执</summary><pre>'+escape(json.dumps(detail,ensure_ascii=False,indent=2))+'</pre></details></section>')
    return ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>神经网络移动前检查</title>
<style>body{margin:0;background:#eef3f4;color:#223b43;font:16px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:30px}p{line-height:1.8}.notice{background:#fff0d8;padding:18px;border-left:4px solid #ba813e}.table{overflow:auto}table{border-collapse:collapse;background:white;font-size:13px;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}section{background:white;margin-top:20px;padding:20px;border-radius:8px}figure{margin:12px 0}img{width:400px;max-width:100%;image-rendering:pixelated}figcaption{line-height:2;color:#536b72}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#176277}</style>
<main><h1>模型判断进入移动前检查</h1><p>实际 TinyFormer 读取同帧 RGB；深度和位姿保留采集时间。处理等待真实推进模拟运动，过期、身份错误或目标检测只增加拒绝；空检测仍需原几何检查。</p><p class="notice">本轮仍使用简化仿真画面。开发探针中模型漏检了三个视角的简化行人；不能把无检测框解释为无人。这里验证接入行为，尚未证明可靠语义避障，也未接通真实航拍的三维定位。</p>
<div class="table"><table><tr><th>场景</th><th>任务终态</th><th>确认格</th><th>视觉判定</th><th>原始原因</th></tr>'''+''.join(rows)+
        '</table></div>'+''.join(cards)+'<p><a href="report.json">完整报告</a> · <a href="protocol.md">固定协议</a> · <a href="manifest.json">文件摘要</a></p></main></html>').encode('utf-8')


def run(output,verify=False):
    base.check_manifest(PARENT)
    parent=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    sources={n:base.sha(ROOT/n) for n in sorted(set(parent['sources'])|set(OWN))}
    if any(sources[n]!=d for n,d in parent['sources'].items()):raise ValueError('parent source changed')
    if base.sha(MODEL)!=MODEL_SHA:raise ValueError('model changed')
    base.check_manifest(base.MODEL.parent)
    model=base.ColorModel.from_dict(json.loads(base.MODEL.read_text(encoding='utf-8')))
    if verify:
        from PIL import Image
        count=base.check_manifest(output)
        report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if report['sources']!=sources or report['parent_sha256']!=base.sha(PARENT/'report.json'):
            raise ValueError('source differs')
        if [c['summary']['spec'] for c in report['cases']]!=specs():raise ValueError('matrix differs')
        with TinyFormerSession(output/'sensor',replay=True) as detector:
            for case in report['cases']:
                raw=json.loads(gzip.decompress((output/case['input']).read_bytes()))
                for c in raw['semantic_checks']:
                    call=c['call']
                    if 'error' in call:continue
                    detector.replay_frame(call)
                    directory=output/'sensor'/call['directory']
                    frame=raw['frames'][c['frame_index']]['frame']
                    with Image.open(directory/'input.png') as im:
                        assert list(im.getdata())==[tuple(p) for p in frame['rgb']]
                        assert prepare(im,call['window'])==gzip.decompress((directory/'input.gz').read_bytes())
                    assert base.sha(directory/'input.png')==call['input_rgb_sha256']
                new,s=simulate(case['summary']['spec'],model,saved=raw)
                if new!=raw or s!=case['summary']:raise ValueError('semantic control replay differs '+case['key'])
                world=base.saved_world(s['world'])
                with base.QuadrotorPhysics(model_xml=base.world_xml(world)) as physics:
                    if base.replay_raw(physics,raw,(3,8))!=s['history'] or physics.state()!=s['actual_final']:
                        raise ValueError('motor replay differs')
                for f in raw['frames']:
                    d=f['frame'];pose=Pose(**{k:tuple(v) for k,v in d['pose'].items()})
                    actual=render_range(world,pose,Intrinsics(**d['intrinsics']),tick=d['tick'],
                        max_range_m=s['spec']['range_m'],invalid=f['sensor_fault_active'])
                    if base.canonical(asdict(actual))!=d:raise ValueError('rerender differs')
                print('verified',case['key'],flush=True)
        if page(report)!=(output/'demo.html').read_bytes():raise ValueError('page differs')
        return dict(verified=True,cases=len(report['cases']),files=count)
    if output.exists():raise FileExistsError('refuse overwrite '+str(output))
    output.mkdir(parents=True)
    cases=[]
    with TinyFormerSession(output/'sensor') as detector:
        for spec in specs():
            raw,s=simulate(spec,model,detector)
            name=spec['key']+'/raw.json.gz'
            base.put(output/name,gzip.compress(json.dumps(raw,separators=(',',':')).encode(),mtime=0))
            cases.append(dict(key=spec['key'],title=spec['title'],input=name,summary=s))
            print(spec['key'],s['result']['state'],s['result']['original_reason'],
                  [c['decision']['reason'] for c in s['semantic_checks']],flush=True)
        startup_s=detector.startup_s
    if sources!={n:base.sha(ROOT/n) for n in sources}:raise ValueError('source changed during run')
    report=dict(kind='semantic-move',sources=sources,parent_sha256=base.sha(PARENT/'report.json'),
                detector_model_sha256=MODEL_SHA,detector_startup_s=startup_s,cases=cases)
    base.dump(output/'report.json',report)
    base.put(output/'protocol.md',(ROOT/'docs/SEMANTIC_MOVE_PROTOCOL.md').read_bytes())
    base.put(output/'demo.html',page(report))
    for name in ('LICENSE','NOTICE','sources.json'):
        base.put(output/'model-source'/name,(MODEL.parent/name).read_bytes())
    base.dump(output/'manifest.json',{p.relative_to(output).as_posix():base.sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases),frames=sum(c['summary']['frames'] for c in cases),
                steps=sum(c['summary']['steps'] for c in cases))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
