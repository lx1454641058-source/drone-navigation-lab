"""来源：本项目原创。检测框开发评价；框覆盖不等于物体检测召回或分割精度。"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import urllib.request

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.realvision import asset_root,validate_assets,sha,dump
from tools.setup_vision import put,digest

ROOT=Path('D:/DroneNavTools/yolox-probe')
REV='6ddff4824372906469a7fae2dc3206c7aa4bbaee'
ASSETS=(
    ('yolox_s.onnx','https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx',35858002,'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'),
    ('LICENSE',f'https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/{REV}/LICENSE',11371,'0ec3668d3274bcf29e8a29e9576d5a2cd96fc78d3c5bec4387355a796e5d9088'),
    ('ONNX-README.md',f'https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/{REV}/demo/ONNXRuntime/README.md',3177,'587068c99b5685a27c8b3460bb28b21a2ac51b3de0a157c9e087241f03d51b35'))


def coverage(truth,boxes,width=960,height=540,source_width=3840,source_height=2160):
    if len(truth)!=width*height:raise ValueError('label shape')
    result={}
    for group,target in [('person',6),('vehicle',3)]:
        mask=bytearray(len(truth));count=0
        for b in boxes:
            if b['group']!=group:continue
            if not all(math.isfinite(b[k]) for k in ['x1','y1','x2','y2']):raise ValueError('nonfinite box')
            if b['x2']<=b['x1'] or b['y2']<=b['y1']:raise ValueError('empty box')
            count+=1
            # 半开区间：只覆盖像素中心实际位于框内的评价格，重叠框只计一次。
            x0=max(0,min(width,math.ceil(b['x1']*width/source_width-.5)))
            x1=max(0,min(width,math.ceil(b['x2']*width/source_width-.5)))
            y0=max(0,min(height,math.ceil(b['y1']*height/source_height-.5)))
            y1=max(0,min(height,math.ceil(b['y2']*height/source_height-.5)))
            for y in range(y0,y1):mask[y*width+x0:y*width+x1]=b'\1'*(x1-x0)
        actual=sum(v==target for v in truth);area=sum(mask)
        hit=sum(bool(v) and t==target for v,t in zip(mask,truth))
        result[group]=dict(boxes=count,label_pixels=actual,covered_label_pixels=hit,box_union_pixels=area,
                           label_coverage=hit/actual if actual else None,
                           target_fraction_in_boxes=hit/area if area else None,frame_fraction=area/len(truth))
    return result


def run(out):
    if out.exists():raise FileExistsError('refuse to overwrite: '+str(out))
    root=asset_root().resolve();lock,runtime=validate_assets(root)
    sources=[]
    for name,url,size,expected in ASSETS:
        file=ROOT/name
        if file.exists():body=file.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=90) as r:body=r.read(size+1)
        if len(body)!=size or digest(body)!=expected:raise ValueError('model resource differs: '+name)
        put(file,body);sources.append(dict(path=name,url=url,bytes=size,sha256=expected))
    cases=[c for c in lock['cases'] if c['split']=='development']
    reference=PROJECT/'work/run-v13-scale-validated'
    manifest=json.loads((reference/'manifest.json').read_text(encoding='utf-8'))
    for c in cases:
        for suffix in ['truth.u8','input.png']:
            rel=c['key']+'/'+suffix
            if sha(reference/rel)!=manifest[rel]:raise ValueError('reference changed')
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'sources.json',sources);dump(out/'runtime-lock.json',runtime)
    put(out/'protocol.md',(PROJECT/'docs/DETECTION_PROBE_PROTOCOL.md').read_bytes())
    worker=Path(__file__).with_suffix('.cjs')
    source_files=[Path(__file__),worker,PROJECT/'drone_nav/detection_math.cjs',PROJECT/'drone_nav/segmentation_tiles.cjs']
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in source_files}
    q=dict(root=str(root),model=str((ROOT/'yolox_s.onnx').resolve()),output=str(out.resolve()),cases=cases)
    dump(out/'request.json',q)
    subprocess.run(['node',str(worker)],input=json.dumps(q),text=True,check=True,timeout=600)
    rows=[]
    for c in cases:
        d=out/c['key'];truth=(reference/c['key']/'truth.u8').read_bytes()
        put(d/'truth.u8',truth);put(d/'input.png',(reference/c['key']/'input.png').read_bytes())
        for grid in [1,4]:
            m=json.loads((d/f'grid{grid}/result.json').read_text(encoding='utf-8'))
            rows.append(dict(key=c['key'],grid=grid,source_width=m['width'],source_height=m['height'],boxes=m['boxes'],
                             coverage=coverage(truth,m['boxes']),inference_ms=sum(t['inference_ms'] for t in m['tiles'])))
    subprocess.run(['node',str(worker)],input=json.dumps(dict(q,replay=True)),text=True,check=True,timeout=600)
    if hashes!={str(p.relative_to(PROJECT)):sha(p) for p in source_files}:raise ValueError('source changed during run')
    report=dict(kind='development_detection_probe',score_threshold=.3,nms_iou=.45,results=rows,sources=hashes,
                protocol_sha256=sha(out/'protocol.md'),replay_passed=True)
    dump(out/'report.json',report)
    template=(PROJECT/'drone_nav/detection_probe.html').read_text(encoding='utf-8')
    put(out/'demo.html',template.replace('__REPORT__',json.dumps(report,ensure_ascii=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{str(p.relative_to(out)).replace('\\','/'):sha(p) for p in out.rglob('*') if p.is_file()})
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description='YOLOX-S 两开发图探针，已有输出拒绝覆盖')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.output),ensure_ascii=False,indent=2))
