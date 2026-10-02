"""来源：本项目原创。真实转换观测的离线地图状态实验及独立格索引核验。"""
import argparse
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN
import html
from itertools import product
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.frame_transform import RigidFrameTransform
from drone_nav.framed_surface_map import FramedSurfaceMap
from drone_nav.realvision import sha,dump
from tools.tum_rgbd_experiment import check_manifest,manifest

OWN=('drone_nav/framed_surface_map.py','tests/test_framed_surface_map.py',
     'tools/framed_surface_map_experiment.py','docs/FRAMED_SURFACE_MAP_PROTOCOL.md')
CAMERA='tum-freiburg3-rgb'
CLOCK='tum-rgbd-relative-seconds'


def reference(parent):
    value=deepcopy(parent['transform'])
    value['rotation']=tuple(tuple(row) for row in value['rotation'])
    value['translation_m']=tuple(value['translation_m'])
    return RigidFrameTransform(**value)


def sources(parent):
    for name,expected in parent['sources'].items():
        if sha(ROOT/name)!=expected: raise ValueError('frozen source differs: '+name)
    return dict(parent['sources'],**{name:sha(ROOT/name) for name in OWN})


def independent_cells(point):
    # Independent arithmetic: decimal floor, not production binary-float division.
    axes=[]
    for coordinate in point:
        scaled=Decimal(str(coordinate))/Decimal('0.25')
        nearest=scaled.to_integral_value(rounding=ROUND_HALF_EVEN)
        axes.append((int(nearest)-1,int(nearest)) if abs(scaled-nearest)<=Decimal('0.00000001')
                    else (int(scaled.to_integral_value(rounding=ROUND_FLOOR)),))
    return list(product(*axes))


def independent_update(expected,finished):
    p=finished['result']['observation']['projection']; grouped={}; count=0
    for observation in p['observations']:
        d=observation['detection']
        for sample in observation['samples']:
            ref=dict(detection_id=d['id'],reported_group=d['group'],pixel=sample['pixel'],
                world_m=sample['world_m'],source_world_m=sample['source_world_m'],association='unverified_box_surface')
            for index in independent_cells(sample['world_m']): grouped.setdefault(index,[]).append(ref)
            count+=1
    for index,refs in grouped.items():
        expected[index]=dict(frame_id=p['frame_id'],camera_id=CAMERA,captured_at_s=p['captured_at_s'],
            valid_until_s=p['valid_until_s'],references=deepcopy(refs),detector_source=p['detector_source'])
    return count,len(grouped)


def audit_snapshot(snapshot,expected):
    now=Decimal(str(snapshot['now_s'])); rows=[]; current=0
    for index,value in sorted(expected.items()):
        valid=now<=Decimal(str(value['valid_until_s']))+Decimal('0.000000001')
        current+=int(valid)
        rows.append(dict(index=list(index),state='current_surface_evidence' if valid else 'unknown_after_expiry',**value))
    if snapshot['cells']!=rows or snapshot['recorded_cells']!=len(rows):
        raise ValueError('independent grid/provenance/time differs')
    if (snapshot['free_cells']!=0 or snapshot['unseen_state']!='unknown_unobserved'
            or snapshot['flight_authorized'] or snapshot['navigation_map_update_allowed'] or snapshot['fault']):
        raise ValueError('map state unexpectedly authorizes or clears space')
    return dict(recorded_cells=len(rows),current_cells=current,expired_unknown_cells=len(rows)-current,free_cells=0)


def negative_cases(first,transform):
    results=[]; now=first['now_s']
    def grid(cap=4096): return FramedSurfaceMap(transform,camera_id=CAMERA,clock_id=CLOCK,max_cells=cap)
    def execute(name,mutate,expected,*,seed=False,when=None,clock=CLOCK,cap=4096,exception=False):
        g=grid(cap)
        if seed: g.ingest(first,now_s=now,clock_id=CLOCK)
        before=deepcopy(g._cells); before_now=g._now
        r=deepcopy(first); mutate(r)
        try:
            answer=g.ingest(r,now_s=now if when is None else when,clock_id=clock)
            reason=answer['reason']; rejected=not answer['accepted'] and reason==expected and not exception
        except ValueError as exc:
            reason=str(exc); rejected=exception and g._now==before_now
        unchanged=g._cells==before
        results.append(dict(name=name,rejected=rejected,cells_unchanged=unchanged,reason=reason))
        if not rejected or not unchanged: raise ValueError('negative case failed: '+name)
        return g
    noop=lambda r:None
    execute('repeated-frame',noop,'REPEATED_FRAME',seed=True)
    def older(r):
        # Explicit injected timestamp fault, not a claim of another real capture.
        m=r['result']; p=m['observation']['projection']
        m['frame_id']=p['frame_id']='injected-older-capture'
        for key in ('captured_at_s','valid_until_s'): m[key]-=.001; p[key]-=.001
        p['age_s']+=.001; r['age_s']+=.001
    execute('injected-older-capture',older,'OUT_OF_ORDER_CAPTURE',seed=True)
    execute('expired-receipt',noop,'EXPIRED_AT_RECEIPT',when=first['result']['valid_until_s']+.001)
    execute('same-name-different-reference',lambda r:r['result']['transform'].update(reference_sha256='b'*64),'REFERENCE_TRANSFORM_MISMATCH')
    execute('wrong-world',lambda r:r['result'].update(world_frame='other-world'),'WORLD_FRAME_MISMATCH')
    execute('wrong-source-clock',lambda r:r['result'].update(clock_id='other-clock'),'SOURCE_CLOCK_MISMATCH')
    execute('wrong-camera',lambda r:r['result']['observation']['projection'].update(camera_id='other-camera'),'CAMERA_MISMATCH')
    execute('conversion-not-ready',noop,'CONVERSION_NOT_READY',when=now-.001)
    def corrupt(r): r['result']['observation']['projection']['observations'][0]['samples'][1]['world_m'][0]+=.1
    execute('modified-second-point',corrupt,None,exception=True)
    g=execute('cell-budget',noop,'CELL_BUDGET_EXCEEDED',cap=1)
    again=g.ingest(first,now_s=now,clock_id=CLOCK)
    if again['reason']!='MAP_FAULT_LATCHED' or g._cells: raise ValueError('capacity fault not latched')
    results[-1]['retry_reason']=again['reason']
    execute('caller-clock-mismatch',noop,None,seed=True,clock='other-clock',exception=True)
    execute('clock-rollback',noop,None,seed=True,when=now-.001,exception=True)
    return results


def page(report):
    snapshots=[r['snapshot'] for r in report['frames']]+[report['expired_snapshot']]
    bounds=[c['index'] for s in snapshots for c in s['cells']]
    x0=min(p[0] for p in bounds)-1; x1=max(p[0] for p in bounds)+2
    z0=min(p[2] for p in bounds)-1; z1=max(p[2] for p in bounds)+2
    def panel(index,snapshot):
        columns={}
        for cell in snapshot['cells']:
            x,y,z=cell['index']; columns.setdefault((x,z),[]).append(cell)
        marks=[]
        for (x,z),cells in sorted(columns.items()):
            active=any(c['state']=='current_surface_evidence' for c in cells)
            px=60+560*(x-x0)/(x1-x0); py=340-280*(z+1-z0)/(z1-z0)
            w=560/(x1-x0); h=280/(z1-z0)
            title=html.escape(f'X/Z 索引 {x}/{z}；Y 层 {[c["index"][1] for c in cells]}；'+('含当前证据' if active else '全部过期未知'))
            marks.append(f'<rect x="{px:.3f}" y="{py:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{"#cf641d" if active else "#a4acb9"}" stroke="white" stroke-width=".7"><title>{title}</title></rect>')
        return f'<svg class="map" data-step="{index}" style="display:{"block" if index==0 else "none"}" viewBox="0 0 680 395" role="img" aria-label="第 {index+1} 阶段三维格子的 X Z 投影"><rect x="60" y="60" width="560" height="280" fill="#f1f3f6"/><path d="M60 60V340H620" stroke="#536478" fill="none"/>{"".join(marks)}<text x="60" y="370">右 X：{x0*.25:.2f} 至 {x1*.25:.2f} 米</text><text x="60" y="35">前 Z：{z0*.25:.2f} 至 {z1*.25:.2f} 米；非地理坐标</text></svg>'
    options=''.join(f'<option value="{i}">{r["offset_s"]} 秒图像 · {r["input_samples"]} 点</option>' for i,r in enumerate(report['frames']))
    stats=[r['audit'] for r in report['frames']]+[report['expired_audit']]
    rows=''.join(f'<tr><td>{r["offset_s"]}</td><td>{r["input_samples"]}</td><td>{r["audit"]["recorded_cells"]}</td><td>{r["audit"]["current_cells"]}</td><td>{r["audit"]["expired_unknown_cells"]}</td></tr>' for r in report['frames'])
    cases=''.join(f'<li>{html.escape(c["name"])}：已拒绝；{html.escape(c["reason"])}</li>' for c in report['negative_cases'])
    details=''.join(f'<details class="detail" data-step="{i}" style="display:{"block" if i==0 else "none"}"><summary>查看这一阶段的逐格来源与期限</summary><pre>{html.escape(json.dumps(s["cells"],ensure_ascii=False,indent=2))}</pre></details>' for i,s in enumerate(snapshots))
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实表面观测的研究地图</title>
<style>body{margin:0;background:#eef2f6;color:#182f43;font:16px/1.75 system-ui;padding:24px}main{max-width:1080px;margin:auto}section{background:white;padding:24px;border-radius:12px;margin:20px 0}h1{font-size:30px}h2{font-size:21px}select{font:inherit;padding:9px;max-width:100%}svg{width:100%;max-height:470px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}pre{overflow:auto;max-height:450px;font-size:13px}.warning{color:#804216}.key{display:flex;gap:22px;flex-wrap:wrap}.dot{display:inline-block;width:14px;height:14px;margin-right:7px}.scroll{overflow:auto}a{color:#165f9d}</style>
<main><h1>真实表面观测的研究地图</h1><p>六帧 · 143 个采样点 · 0.25 米格宽 · 保存时间的离线重放</p>
<section><h2>哪些地方有证据，哪些仍然未知</h2><p class="warning">仅记录检测框内表面点，可能含背景。没有空闲证明、导航坐标标定或飞行授权。</p>
<label for="step">观察时刻：</label><select id="step">'''+options+'''<option value="6">末帧期限后 · 全部过期</option></select><p id="stats"></p>
<div class="key"><span><i class="dot" style="background:#cf641d"></i>含当前表面证据</span><span><i class="dot" style="background:#a4acb9"></i>记录已过期，未知</span><span><i class="dot" style="background:#f1f3f6"></i>投影无记录，未知</span></div>
'''+''.join(panel(i,s) for i,s in enumerate(snapshots))+'''<p>图中合并显示 X/Z 相同、Y 不同的格子，任一层有当前证据即为橙色；不是完整三维占据图。鼠标悬停可看 Y 层。背景和灰色都不代表可通行。6 秒图像没有样本，也不清除或延长已有证据。</p>'''+details+'''</section>
<section><h2>逐帧状态</h2><div class="scroll"><table><tr><th>图像偏移 / 秒</th><th>输入点</th><th>累计有记录格</th><th>当前证据格</th><th>过期未知格</th></tr>'''+rows+'''</table></div><p>点数与格数不是物体数量；相同格只保存最近一帧来源。不同时间快照单独保存。</p></section>
<section><h2>十二项拒绝检查</h2><p>以下为固定注入的错误条件，与上方真实六帧顺序重放分开报告。</p><ul>'''+cases+'''</ul></section><section><h2>证据与限制</h2><p>六帧沿用原转换完成时间，不新增模型推理、不测本阶段在线耗时。独立十进制格索引与逐格来源/期限核验通过，不代表识别精度或导航安全。</p><p><a href="report.json">完整报告</a> · <a href="protocol.md">固定方案</a> · <a href="dataset-sources.json">数据署名与许可</a></p></section></main>
<script>const stats='''+json.dumps(stats)+''';const select=document.getElementById('step');function show(){const i=Number(select.value);document.querySelectorAll('[data-step]').forEach(e=>e.style.display=Number(e.dataset.step)===i?'block':'none');const s=stats[i];document.getElementById('stats').textContent=`有记录 ${s.recorded_cells} 格 · 当前证据 ${s.current_cells} 格 · 过期未知 ${s.expired_unknown_cells} 格 · 已证明空闲 0 格`;}select.addEventListener('change',show);show();</script></html>'''


def run(parent_dir,output,verify=False):
    check_manifest(parent_dir)
    parent=json.loads((parent_dir/'report.json').read_text(encoding='utf-8'))
    transform=reference(parent); source_hashes=sources(parent)
    if [r['offset_s'] for r in parent['frames']]!=[2,4,6,8,10,12]: raise ValueError('fixed frames differ')
    untouched=deepcopy(parent)
    report=dict(sources=source_hashes,parent_sha256=sha(parent_dir/'report.json'),
        transform=parent['transform'],timing_scope='offline_saved_conversion_completion_times',
        new_model_calls=0,physical_flights=0,navigation_map_updates=0,frames=[])
    grid=FramedSurfaceMap(transform,camera_id=CAMERA,clock_id=CLOCK)
    expected={}
    for row,required in zip(parent['frames'],(48,28,0,37,15,15)):
        finished=row['finished']; now=finished['now_s']; old=deepcopy(expected)
        answer=grid.ingest(finished,now_s=now,clock_id=CLOCK)
        if not answer['accepted']: raise ValueError('fixed observation not accepted: '+answer['reason'])
        samples,updated=independent_update(expected,finished)
        if samples!=required or answer['input_samples']!=samples or answer['updated_cells']!=updated:
            raise ValueError('input/cell counts differ')
        if samples==0 and expected!=old: raise ValueError('empty frame changed evidence')
        snapshot=grid.snapshot(now_s=now,clock_id=CLOCK); audit=audit_snapshot(snapshot,expected)
        report['frames'].append(dict(offset_s=row['offset_s'],input_samples=samples,answer=answer,snapshot=snapshot,audit=audit))
    final_now=parent['frames'][-1]['finished']['result']['valid_until_s']+.001
    report['expired_snapshot']=grid.snapshot(now_s=final_now,clock_id=CLOCK)
    report['expired_audit']=audit_snapshot(report['expired_snapshot'],expected)
    if report['expired_audit']['current_cells']!=0: raise ValueError('evidence survived expiry')
    report['negative_cases']=negative_cases(parent['frames'][0]['finished'],transform)
    if parent!=untouched or sources(parent)!=source_hashes: raise ValueError('input/source changed')
    if verify:
        count=check_manifest(output)
        saved=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if report!=saved or page(report)!=(output/'demo.html').read_text(encoding='utf-8'):
            raise ValueError('map replay/report/page differs')
        if (output/'protocol.md').read_bytes()!=(ROOT/'docs/FRAMED_SURFACE_MAP_PROTOCOL.md').read_bytes():
            raise ValueError('protocol differs')
        if (output/'dataset-sources.json').read_bytes()!=(parent_dir/'dataset-sources.json').read_bytes():
            raise ValueError('dataset attribution differs')
        if check_manifest(output)!=count: raise ValueError('archive changed')
    else:
        output.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(ROOT/'docs/FRAMED_SURFACE_MAP_PROTOCOL.md',output/'protocol.md')
        shutil.copyfile(parent_dir/'dataset-sources.json',output/'dataset-sources.json')
        dump(output/'report.json',report)
        (output/'demo.html').write_text(page(report),encoding='utf-8')
        dump(output/'manifest.json',manifest(output)); count=check_manifest(output)
    return dict(verified=verify,frames=6,input_samples=143,negative_cases=len(report['negative_cases']),
        snapshots=7,final=report['expired_audit'],files=count,sources=len(source_hashes),
        stages=[r['audit'] for r in report['frames']])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,default=ROOT/'work/frame-transform-01')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.parent,args.output,args.verify),indent=2),flush=True)
