"""来源：本项目原创。固定整图去重对照和按文件名前缀隔离的保留图片验证。"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import zipfile

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.realvision import sha,dump,asset_root,validate_assets
from drone_nav.object_metrics import evaluate_frame,aggregate,parse_visdrone
from drone_nav.detection_failures import changes
from tools.object_probe import check_manifest,MODEL as YOLOX_MODEL,MODEL_SHA as YOLOX_SHA
from tools.tinyformer_probe import prepare,MODEL as TINY_MODEL,MODEL_SHA as TINY_SHA
from tools.setup_object_data import ARCHIVE_SHA
from tools.setup_vision import put

SOURCE=PROJECT/'work/tinyformer-probe-01'
ARCHIVE=Path('D:/DroneNavTools/visdrone-eval/mirrored-VisDrone2019-DET-val.zip')
SOURCES=('tools/whole_detection_study.py','tools/whole_detection_worker.cjs','tools/tinyformer_probe.py',
         'drone_nav/tinyformer_math.cjs','drone_nav/detection_math.cjs','drone_nav/object_metrics.py',
         'drone_nav/detection_failures.py','drone_nav/whole_detection_lab.html')


def select_reserved(names,old_selected):
    images=sorted(n for n in names if n.startswith('VisDrone2019-DET-val/images/') and n.endswith('.jpg'))
    if len(images)!=548 or len(set(images))!=548:raise ValueError('validation image inventory differs')
    groups={}
    for name in images:groups.setdefault(Path(name).stem.split('_')[0],name)
    order=sorted(groups.values(),key=lambda n:hashlib.sha256(('drone-nav-object-dev-v1:'+Path(n).name).encode()).hexdigest())
    if order[:12]!=old_selected:raise ValueError('original development selection differs')
    selected=order[12:24]
    if len(selected)!=12 or {Path(n).stem.split('_')[0] for n in selected}&{Path(n).stem.split('_')[0] for n in old_selected}:
        raise ValueError('selection overlaps')
    return selected


def worker(q):
    subprocess.run(['node',str(PROJECT/'tools/whole_detection_worker.cjs')],input=json.dumps(q),text=True,check=True,timeout=600)


def evaluate(cases,rows):
    for r in rows:
        c=next(c for c in cases if c['key']==r['key'])
        r['evaluations']={t:evaluate_frame(c['annotations'],r['boxes'],c['width'],c['height'],float(t)) for t in ('0.50','0.75')}
    arms=tuple(dict.fromkeys(r['arm'] for r in rows))
    summary={arm:{t:aggregate([r['evaluations'][t] for r in rows if r['arm']==arm]) for t in ('0.50','0.75')} for arm in arms}
    transitions=[]
    for c in cases:
        a=next(r for r in rows if r['key']==c['key'] and r['arm']=='tinyformer-1')
        b=next(r for r in rows if r['key']==c['key'] and r['arm']=='tinyformer-nms')
        transitions.append(dict(key=c['key'],thresholds={t:{g:changes(a['evaluations'][t][g],b['evaluations'][t][g]) for g in ('person','vehicle')} for t in ('0.50','0.75')}))
    return dict(cases=cases,results=rows,summary=summary,transitions=transitions)


def compute(out,phase,base):
    if phase=='dev':
        cases=base['analysis']['cases'];rows=[]
        for c in cases:
            original=next(r for r in base['analysis']['results'] if r['key']==c['key'] and r['arm']=='tinyformer-1')
            clean=json.loads((out/(c['key']+'.json')).read_text(encoding='utf-8'))
            rows.extend([dict(key=c['key'],arm='tinyformer-1',boxes=original['boxes'],inference_ms=original['inference_ms']),
                         dict(key=c['key'],arm='tinyformer-nms',boxes=clean['boxes'],inference_ms=original['inference_ms'])])
    else:
        cases=json.loads((out/'data.json').read_text(encoding='utf-8'))['cases'];rows=[]
        for c in cases:
            raw=json.loads((out/c['key']/'result.json').read_text(encoding='utf-8'));m=raw['measurements'];d=raw['decoded']
            for arm,key in [('yolox-1','yolox'),('tinyformer-1','tinyformer'),('tinyformer-nms','tinyformer_nms')]:
                boxes=d[key]['boxes'] if key=='tinyformer_nms' else d[key]
                rows.append(dict(key=c['key'],arm=arm,boxes=boxes,inference_ms=m['yolox_inference_ms'] if key=='yolox' else m['tinyformer_inference_ms']))
    return evaluate(cases,rows)


def run(out,phase,verify=False):
    if not verify and out.exists():raise FileExistsError('refuse to overwrite '+str(out))
    check_manifest(SOURCE);base=json.loads((SOURCE/'report.json').read_text(encoding='utf-8'))
    for name,digest in base['sources'].items():
        if sha(PROJECT/name)!=digest:raise ValueError('candidate source differs')
    hashes={n:sha(PROJECT/n) for n in SOURCES}
    if verify:
        count=check_manifest(out);report=json.loads((out/'report.json').read_text(encoding='utf-8'))
        if report['phase']!=phase or report['source_sha256']!=sha(SOURCE/'report.json') or report['sources']!=hashes or report['protocol_sha256']!=sha(out/'protocol.md'):
            raise ValueError('source or protocol differs')
        q=json.loads((out/'request.json').read_text(encoding='utf-8'));q.update(output=str(out.resolve()),replay=True);worker(q)
        if report['analysis']!=compute(out,phase,base):raise ValueError('evaluation replay differs')
        return dict(verified=True,phase=phase,files=count,evaluation_groups=96 if phase=='dev' else 144)
    if phase=='holdout' and (sha(ARCHIVE)!=ARCHIVE_SHA or sha(YOLOX_MODEL)!=YOLOX_SHA or sha(TINY_MODEL)!=TINY_SHA):raise ValueError('model or dataset hash differs')
    out.mkdir(parents=True,exist_ok=False)
    put(out/'protocol.md',(PROJECT/'docs/WHOLE_DEDUP_PROTOCOL.md').read_bytes())
    for name in ('sources.json','LICENSE','NOTICE','README.md'):put(out/'model-source'/name,(SOURCE/'model-source'/name).read_bytes())
    if phase=='dev':
        cases=base['analysis']['cases']
        q=dict(dev=True,cases=cases,source=str(SOURCE.resolve()),output=str(out.resolve()))
        for c in cases:put(out/c['key']/'input.png',(SOURCE/c['key']/'input.png').read_bytes())
        put(out/'selection.json',(SOURCE/'selection.json').read_bytes())
    else:
        from PIL import Image,__version__ as pillow_version
        previous=json.loads((SOURCE/'selection.json').read_text(encoding='utf-8'))
        with zipfile.ZipFile(ARCHIVE) as z:
            names=z.namelist();selected=select_reserved(names,previous['selected'])
            # 选择只依赖文件名；此文件写完后才读取图像或标注。
            dump(out/'selection.json',dict(selected=selected,excluded_development=previous['selected'],inventory=names,
                  rule='original deterministic ordering, positions 13 to 24',archive_sha256=ARCHIVE_SHA,development_selection_sha256=sha(SOURCE/'selection.json')))
            cases=[];times=[]
            for name in selected:
                key=Path(name).stem;raw=z.read(name);labels=z.read('VisDrone2019-DET-val/annotations/'+key+'.txt')
                put(out/key/'original.jpg',raw);put(out/key/'annotations.txt',labels)
                with Image.open(io.BytesIO(raw)) as im:
                    rgb=im.convert('RGB');w,h=rgb.size;png=io.BytesIO();rgb.save(png,format='PNG')
                    with Image.open(io.BytesIO(png.getvalue())) as decoded:
                        if decoded.tobytes()!=rgb.tobytes():raise ValueError('PNG roundtrip')
                    put(out/key/'input.png',png.getvalue())
                    start=perf_counter();tensor=prepare(rgb,dict(x0=0,y0=0,width=w,height=h));elapsed=(perf_counter()-start)*1000
                    put(out/key/'tiny-input.gz',gzip.compress(tensor,mtime=0));times.append(dict(key=key,tinyformer_preprocess_ms=elapsed))
                cases.append(dict(key=key,width=w,height=h,annotations=parse_visdrone(labels.decode('utf-8-sig'),w,h)))
        dump(out/'data.json',dict(cases=cases,pillow=pillow_version,preprocessing=times))
        runtime=asset_root().resolve();_,lock=validate_assets(runtime);dump(out/'runtime-lock.json',lock)
        q=dict(cases=[{k:c[k] for k in ('key','width','height')} for c in cases],models=dict(yolox=str(YOLOX_MODEL),tinyformer=str(TINY_MODEL)),
               runtime=str(runtime),output=str(out.resolve()))
    dump(out/'request.json',q);worker(q);worker(dict(q,replay=True))
    analysis=compute(out,phase,base)
    if hashes!={n:sha(PROJECT/n) for n in SOURCES}:raise ValueError('source changed during run')
    report=dict(phase=phase,sources=hashes,source_sha256=sha(SOURCE/'report.json'),protocol_sha256=sha(out/'protocol.md'),
                models=dict(yolox=YOLOX_SHA,tinyformer=TINY_SHA),analysis=analysis)
    dump(out/'report.json',report)
    template=(PROJECT/'drone_nav/whole_detection_lab.html').read_text(encoding='utf-8')
    put(out/'demo.html',template.replace('__DATA__',json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return analysis['summary']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=('dev','holdout'),required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.phase,a.verify),ensure_ascii=False,indent=2))
