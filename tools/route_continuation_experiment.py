"""来源：本项目原创。固定航向/转弯矩阵与连续观测缓存的物理任务验证。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
from html import escape
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.continuation_history import ContinuationHistory
from drone_nav.classifier import ColorModel
from drone_nav.mission_supervisor import MissionSupervisor
from drone_nav.native_physics import QuadrotorPhysics
from drone_nav.physical_vehicle import world_xml
from drone_nav.verify_coupled import saved_world
from drone_nav.verify_descent import replay_raw
from drone_nav.pinhole import Pose,Intrinsics
from drone_nav.range_observation import render_range
from drone_nav.realvision import sha,dump
from tools.range_observation_experiment import Vehicle as RangeVehicle,specs as old_specs,MODEL,canonical
from tools.object_probe import check_manifest
from tools.setup_vision import put

PARENT=ROOT/'work/range-observation-01'
OWN=('drone_nav/continuation_history.py','tools/route_continuation_experiment.py',
     'tests/test_continuation_history.py','docs/ROUTE_CONTINUATION_PROTOCOL.md')


def specs():
    base=next(s for s in old_specs() if s['key']=='open')
    cases=[('old-east','原缓存：向东四段',(7,8),False),
           ('east','新缓存：向东四段',(7,8),True),
           ('north','向北两段',(3,10),True),('south','向南两段',(3,6),True),
           ('west','向西两段',(1,8),True),('turn-north','向东北：四段及转弯',(5,10),True),
           ('turn-south','向东南：四段及转弯',(5,6),True)]
    return [dict(base,key=k,title=t,goal=list(g),continuation=h,max_ticks=10) for k,t,g,h in cases]


class Vehicle(RangeVehicle):
    def __init__(self,s,saved=None):
        super().__init__(s,saved)
        if s['continuation']:
            old=self.sampling_history
            self.sampling_history=ContinuationHistory(divisions=old.divisions,initial_region=old.initial_region,
                horizon_s=self.budget.max_move_s+.5,free_ttl_s=self.budget.free_ttl_s)


def simulate(s,model,saved=None):
    v=Vehicle(s,saved)
    try:
        result=MissionSupervisor(v,model,(3,8),tuple(s['goal']),max_ticks=s['max_ticks']).run()
        if v.history[-1]!=v.state():v.history.append(v.state())
        frames=v.frames if saved is None else saved['frames']
        if saved is not None and (v.frame_index!=len(frames) or v.contact_index!=len(saved['contact_readings'])):
            raise ValueError('unused route inputs')
        raw=dict(frames=frames,commands=v.commands,contact_readings=v.contact_readings)
        h=v.sampling_history
        summary=dict(spec=s,result=result,world=asdict(v.world),history=v.history,actual_final=v.state(),
            frames=len(frames),steps=v.steps,actions=v.actions,audit=v.audit,move_checks=v.move_checks,
            source_frames=v.source_frames,probes=v.probes,range_outcomes=dict(Counter(x for item in frames for x in item['frame']['outcomes'])),
            initial_region=asdict(h.initial_region),dll_sha256=v.physics.dll_sha256,model_sha256=v.physics.model_sha256,
            cache=dict(retired=getattr(h,'retired',[]),peak_frames=getattr(h,'peak_frames',None),final_frames=len(h._entries)))
        return canonical(raw),canonical(summary)
    finally:v.close()


def page(report):
    rows=[];cards=[]
    for c in report['cases']:
        s=c['summary'];r=s['result'];trace=r['navigation']['trace'] if r['navigation'] else []
        cells=[r['start']]+[t['position'] for t in trace if t['receipt'] and t['receipt']['completed']]
        values=(c['title'],r['state'],str(r['confirmed_cell']),len(cells)-1,
                s['cache']['peak_frames'] if s['cache']['peak_frames'] is not None else '未记录',r['original_reason'] or '接地确认')
        rows.append('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in values)+'</tr>')
        xy=lambda p:f'{p[0]*28:.2f},{(16-p[1])*28:.2f}'
        actual=' '.join(xy(p['position']) for p in s['history'][::5])
        path=' '.join(xy((p[0]+.5,p[1]+.5)) for p in cells)
        gx,gy=s['spec']['goal'];tx,ty=(gx+.5)*28,(16-gy-.5)*28
        svg=f'<svg viewBox="0 0 560 448" role="img" aria-label="{escape(c["title"])}的俯视轨迹"><rect x="0" y="0" width="560" height="448" fill="#f2f6f7"/><polyline points="{path}" fill="none" stroke="#bbc9cb" stroke-width="6"/><polyline points="{actual}" fill="none" stroke="#166d76" stroke-width="2"/><circle cx="98" cy="210" r="5" fill="#277747"/><circle cx="{tx}" cy="{ty}" r="7" fill="none" stroke="#c56b24" stroke-width="2"/></svg>'
        detail=dict(confirmed_route=cells,frames=s['frames'],steps=s['steps'],cache_peak=s['cache']['peak_frames'],
                    retired_frames=len(s['cache']['retired']),audit=s['audit'],probe_states=[p['status'] for p in s['probes']])
        cards.append('<section><h2>'+escape(c['title'])+'</h2>'+svg+'<p>绿点为起点，橙圈为目标；青线为实际俯视轨迹，灰线为已确认路线。仅保存结果。</p><details><summary>路线、缓存及失败详情</summary><pre>'+escape(json.dumps(detail,ensure_ascii=False,indent=2))+'</pre></details></section>')
    return ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>长路线与转弯验证</title><style>body{margin:0;background:#eef3f4;color:#203d48;font:16px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;padding:26px;margin:auto}p{line-height:1.8}.notice{background:#fff1dc;padding:16px;border-left:4px solid #b77a32}.table{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}td,th{padding:12px;border-bottom:1px solid #dce5e8;text-align:left}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}section{background:white;padding:18px;margin-top:18px}h2{font-size:20px}svg{width:100%;max-height:300px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}a{color:#166d76}</style><main><h1>从两段直线到连续路线与转弯</h1><p>原方法在第二次主动补拍时达到 128 帧上限。新方法只移除已无法覆盖下一段最早停止期限的图像，保留地图障碍、原时间与全部输入归档。</p><p class="notice">仍为理想传感器、精确位姿、静态无障碍场地和明确起始声明下的开发实验。方向或路线通过不代表解决细障碍、动态避障或真实配送；中止与失败一并保留。</p><div class="table"><table><thead><tr><th>场景</th><th>终态</th><th>确认格</th><th>完成段数</th><th>缓存峰值</th><th>原因</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div><div class="cards">'''+''.join(cards)+'''</div><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定协议</a> · <a href="manifest.json">文件摘要</a></p></main></html>''').encode('utf-8')


def archive_case(spec,model,output):
    raw,s=simulate(spec,model);name=spec['key']+'/raw.json.gz'
    put(output/name,gzip.compress(json.dumps(raw,separators=(',',':')).encode(),mtime=0))
    return dict(key=spec['key'],title=spec['title'],input=name,summary=s)


def verify_case(c,model,output):
    raw=json.loads(gzip.decompress((output/c['input']).read_bytes()))
    new,s=simulate(c['summary']['spec'],model,raw)
    if new!=raw or s!=c['summary']:raise ValueError('route control replay differs '+c['key'])
    world=saved_world(s['world'])
    with QuadrotorPhysics(model_xml=world_xml(world)) as physics:
        if replay_raw(physics,raw,(3,8))!=s['history'] or physics.state()!=s['actual_final']:raise ValueError('motor replay differs')
    for item in raw['frames']:
        d=item['frame'];p=Pose(**{k:tuple(v) for k,v in d['pose'].items()})
        f=render_range(world,p,Intrinsics(**d['intrinsics']),tick=d['tick'],max_range_m=s['spec']['range_m'])
        if canonical(asdict(f))!=d:raise ValueError('route image rerender differs')
    return c['key']


def run(output,verify=False):
    check_manifest(PARENT);parent=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    hashes={n:sha(ROOT/n) for n in sorted(set(parent['sources'])|set(OWN))}
    if any(hashes[n]!=d for n,d in parent['sources'].items()):raise ValueError('parent source changed')
    check_manifest(MODEL.parent);model=ColorModel.from_dict(json.loads(MODEL.read_text(encoding='utf-8')))
    if verify:
        files=check_manifest(output);report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if report['sources']!=hashes or report['parent_sha256']!=sha(PARENT/'report.json'):raise ValueError('source/input mismatch')
        if [c['summary']['spec'] for c in report['cases']]!=specs():raise ValueError('route matrix mismatch')
        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(verify_case,c,model,output) for c in report['cases']]
            for job in as_completed(jobs):print('verified',job.result(),flush=True)
        if page(report)!=(output/'demo.html').read_bytes():raise ValueError('page differs')
        return dict(verified=True,files=files,cases=len(report['cases']))
    if output.exists():raise FileExistsError('refuse overwrite '+str(output))
    output.mkdir(parents=True);found={}
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(archive_case,spec,model,output) for spec in specs()]
        for job in as_completed(jobs):
            c=job.result();found[c['key']]=c;s=c['summary']
            print(c['key'],s['result']['state'],s['result']['original_reason'],s['result']['confirmed_cell'],flush=True)
    cases=[found[s['key']] for s in specs()]
    if hashes!={n:sha(ROOT/n) for n in hashes}:raise ValueError('sources changed during run')
    report=dict(kind='route-continuation',sources=hashes,parent_sha256=sha(PARENT/'report.json'),cases=cases)
    dump(output/'report.json',report);put(output/'demo.html',page(report));put(output/'protocol.md',(ROOT/'docs/ROUTE_CONTINUATION_PROTOCOL.md').read_bytes())
    dump(output/'manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(cases=len(cases),frames=sum(c['summary']['frames'] for c in cases),steps=sum(c['summary']['steps'] for c in cases))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
