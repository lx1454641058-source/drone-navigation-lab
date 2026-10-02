"""来源：本项目原创。位置候选合并的开发重放和第三批保留图片验证。"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from tools import whole_detection_study as whole
from drone_nav.query_dedup import merge_queries
from drone_nav.object_metrics import parse_visdrone
from drone_nav.detection_failures import changes
from drone_nav.realvision import dump,sha,asset_root,validate_assets
from tools.setup_vision import put

OLD_A=PROJECT/'work/whole-dedup-dev-01'
OLD_B=PROJECT/'work/whole-dedup-holdout-01'
SOURCES=tuple(dict.fromkeys((*whole.SOURCES,'tools/query_detection_study.py','drone_nav/query_dedup.py','drone_nav/query_detection_lab.html')))
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def select_third(names,first,second):
    if whole.select_reserved(names,first)!=second:raise ValueError('second selection differs')
    representatives={}
    for n in sorted(names):
        if n.startswith('VisDrone2019-DET-val/images/') and n.endswith('.jpg'):
            representatives.setdefault(Path(n).stem.split('_')[0],n)
    ordered=sorted(representatives.values(),key=lambda n:hashlib.sha256(('drone-nav-object-dev-v1:'+Path(n).name).encode()).hexdigest())
    selected=ordered[24:36]
    if len(selected)!=12 or len({Path(n).stem.split('_')[0] for n in first+second+selected})!=36:
        raise ValueError('third selection missing or overlaps')
    return selected


def evaluate(cases,rows):
    result=whole.evaluate(cases,rows)
    result['query_transitions']=[]
    for c in cases:
        a=next(r for r in rows if r['key']==c['key'] and r['arm']=='tinyformer-1')
        b=next(r for r in rows if r['key']==c['key'] and r['arm']=='tinyformer-query')
        result['query_transitions'].append(dict(key=c['key'],thresholds={t:{g:changes(a['evaluations'][t][g],b['evaluations'][t][g]) for g in ('person','vehicle')} for t in ('0.50','0.75')}))
    return result


def originals(out,phase):
    if phase=='holdout':
        return whole.compute(out,'holdout',None)
    parent=OLD_A if phase=='dev-a' else OLD_B
    a=read(parent/'report.json')['analysis'];rows=[dict(r) for r in a['results']]
    if phase=='dev-a':
        baseline=read(whole.SOURCE/'report.json')['analysis']
        rows.extend(dict(r) for r in baseline['results'] if r['arm']=='yolox-1')
    # 每张图片展示同样四个方案顺序；不混合两批开发图的统计。
    rows.sort(key=lambda r:(r['key'],('yolox-1','tinyformer-1','tinyformer-nms').index(r['arm'])))
    return dict(cases=a['cases'],results=rows)


def compute(out,phase,write=False):
    base=originals(out,phase);rows=base['results']
    for c in base['cases']:
        row=next(r for r in rows if r['key']==c['key'] and r['arm']=='tinyformer-1')
        start=perf_counter();merged=merge_queries(row['boxes']);elapsed=(perf_counter()-start)*1000
        p=out/c['key']/'query.json'
        if write:dump(p,dict(result=merged,processing_ms=elapsed))
        elif read(p)['result']!=merged:raise ValueError('query merge replay differs')
        rows.append(dict(key=c['key'],arm='tinyformer-query',boxes=merged['boxes'],inference_ms=row['inference_ms']))
    return evaluate(base['cases'],rows)


def prepare_reserved(out):
    from PIL import Image,__version__ as pillow_version
    if whole.sha(whole.ARCHIVE)!=whole.ARCHIVE_SHA or sha(whole.YOLOX_MODEL)!=whole.YOLOX_SHA or sha(whole.TINY_MODEL)!=whole.TINY_SHA:
        raise ValueError('model or archive differs')
    first=read(OLD_A/'selection.json')['selected'];second=read(OLD_B/'selection.json')['selected']
    with zipfile.ZipFile(whole.ARCHIVE) as z:
        names=z.namelist();selected=select_third(names,first,second)
        dump(out/'selection.json',dict(selected=selected,excluded_development=first+second,inventory=names,
             rule='original deterministic ordering, positions 25 to 36',archive_sha256=whole.ARCHIVE_SHA))
        cases=[];times=[]
        for n in selected:
            key=Path(n).stem;raw=z.read(n);labels=z.read('VisDrone2019-DET-val/annotations/'+key+'.txt')
            put(out/key/'original.jpg',raw);put(out/key/'annotations.txt',labels)
            with Image.open(io.BytesIO(raw)) as im:
                rgb=im.convert('RGB');w,h=rgb.size;png=io.BytesIO();rgb.save(png,format='PNG')
                with Image.open(io.BytesIO(png.getvalue())) as test:
                    if test.tobytes()!=rgb.tobytes():raise ValueError('PNG pixels differ')
                put(out/key/'input.png',png.getvalue())
                start=perf_counter();tensor=whole.prepare(rgb,dict(x0=0,y0=0,width=w,height=h));elapsed=(perf_counter()-start)*1000
                put(out/key/'tiny-input.gz',gzip.compress(tensor,mtime=0));times.append(dict(key=key,tinyformer_preprocess_ms=elapsed))
            cases.append(dict(key=key,width=w,height=h,annotations=parse_visdrone(labels.decode('utf-8-sig'),w,h)))
    dump(out/'data.json',dict(cases=cases,pillow=pillow_version,preprocessing=times))
    runtime=asset_root().resolve();_,lock=validate_assets(runtime);dump(out/'runtime-lock.json',lock)
    q=dict(cases=[{k:c[k] for k in ('key','width','height')} for c in cases],models=dict(yolox=str(whole.YOLOX_MODEL),tinyformer=str(whole.TINY_MODEL)),runtime=str(runtime),output=str(out.resolve()))
    dump(out/'request.json',q);whole.worker(q);whole.worker(dict(q,replay=True))


def run(out,phase,verify=False):
    if not verify and out.exists():raise FileExistsError('refuse to overwrite '+str(out))
    # 原归档及其源码仍须能重放，避免在过期/损坏的基线上比较。
    whole.run(OLD_A,'dev',True);whole.run(OLD_B,'holdout',True)
    hashes={n:sha(PROJECT/n) for n in SOURCES}
    parents={str(p.relative_to(PROJECT)):sha(p/'report.json') for p in (OLD_A,OLD_B,whole.SOURCE)}
    if verify:
        count=whole.check_manifest(out);r=read(out/'report.json')
        if r['phase']!=phase or r['sources']!=hashes or r['parents']!=parents or r['protocol_sha256']!=sha(out/'protocol.md'):
            raise ValueError('source, parents or protocol differs')
        if phase=='holdout':
            q=read(out/'request.json');q.update(output=str(out.resolve()),replay=True);whole.worker(q)
        if r['analysis']!=compute(out,phase):raise ValueError('metric replay differs')
        return dict(verified=True,phase=phase,files=count,evaluation_groups=192)
    out.mkdir(parents=True,exist_ok=False)
    put(out/'protocol.md',(PROJECT/'docs/QUERY_DEDUP_PROTOCOL.md').read_bytes())
    for name in ('LICENSE','NOTICE','README.md','sources.json'):put(out/'model-source'/name,(whole.SOURCE/'model-source'/name).read_bytes())
    if phase=='holdout':prepare_reserved(out)
    else:
        parent=OLD_A if phase=='dev-a' else OLD_B
        for c in originals(out,phase)['cases']:put(out/c['key']/'input.png',(parent/c['key']/'input.png').read_bytes())
        put(out/'selection.json',(parent/'selection.json').read_bytes())
    analysis=compute(out,phase,write=True)
    if hashes!={n:sha(PROJECT/n) for n in SOURCES}:raise ValueError('source changed while running')
    report=dict(phase=phase,sources=hashes,parents=parents,protocol_sha256=sha(out/'protocol.md'),
                models=dict(yolox=whole.YOLOX_SHA,tinyformer=whole.TINY_SHA),analysis=analysis)
    dump(out/'report.json',report)
    template=(PROJECT/'drone_nav/query_detection_lab.html').read_text(encoding='utf-8')
    put(out/'demo.html',template.replace('__DATA__',json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return analysis['summary']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--phase',required=True,choices=('dev-a','dev-b','holdout'));p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.output,a.phase,a.verify),ensure_ascii=False,indent=2))
