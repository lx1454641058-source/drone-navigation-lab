"""来源：本项目原创。固定航拍模型的十二图开发对照，独立于飞行控制。"""
import argparse
from array import array
import gzip
import json
from pathlib import Path
import struct
import subprocess
import sys
from time import perf_counter

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.realvision import sha,dump,asset_root,validate_assets
from drone_nav.object_metrics import evaluate_frame,aggregate
from tools.object_probe import check_manifest
from tools.setup_vision import put

ROOT=Path('D:/DroneNavTools/tinyformer-probe')
MODEL=ROOT/'LibreTinyFormers-visdrone.onnx'
MODEL_SHA='0cd550196b1a69fd68a0993aa8df62e6f52346dbb970735dd57957151917d509'
SOURCE_NAMES=('tools/tinyformer_probe.py','tools/tinyformer_worker.cjs','drone_nav/tinyformer_math.cjs',
              'drone_nav/detection_math.cjs','drone_nav/object_metrics.py','drone_nav/tinyformer_lab.html')


def f32(x):
    return struct.unpack('<f',struct.pack('<f',x))[0]


def prepare(image,window):
    """Pillow 负责与发布者相同的缩放；查表以逐步 float32 计算 RGB 归一化。"""
    from PIL import Image
    x,y,w,h=(window[k] for k in ('x0','y0','width','height'))
    if any(type(v) is not int for v in (x,y,w,h)) or min(x,y)<0 or min(w,h)<=0 or x+w>image.width or y+h>image.height:
        raise ValueError('crop outside image')
    raw=image.convert('RGB').crop((x,y,x+w,y+h)).resize((640,640),Image.Resampling.BILINEAR).tobytes()
    values=array('f')
    for c,(mean,std) in enumerate(zip((.485,.456,.406),(.229,.224,.225))):
        mean,std=f32(mean),f32(std)
        lut=[f32(f32(f32(v/255)-mean)/std) for v in range(256)]
        values.extend(lut[v] for v in raw[c::3])
    if sys.byteorder!='little':values.byteswap()
    return values.tobytes()


def worker(q):
    subprocess.run(['node',str(PROJECT/'tools/tinyformer_worker.cjs')],input=json.dumps(q),text=True,check=True,timeout=1800)


def compute(out,base):
    rows=[]
    for case in base['cases']:
        key,w,h=case['key'],case['width'],case['height']
        for grid in (1,4):
            old=next(r for r in base['results'] if r['key']==key and r['grid']==grid)
            rows.append(dict(key=key,arm=f'yolox-{grid}',grid=grid,boxes=old['boxes'],evaluations=old['evaluations'],inference_ms=old['inference_ms']))
            raw=json.loads((out/key/f'grid{grid}/result.json').read_text(encoding='utf-8'))
            evaluation={t:evaluate_frame(case['annotations'],raw['boxes'],w,h,float(t)) for t in ('0.50','0.75')}
            rows.append(dict(key=key,arm=f'tinyformer-{grid}',grid=grid,boxes=raw['boxes'],evaluations=evaluation,
                             degenerate=raw['degenerate'],inference_ms=sum(t['inference_ms'] for t in raw['tiles'])))
    arms=('yolox-1','yolox-4','tinyformer-1','tinyformer-4')
    summary={arm:{t:aggregate([r['evaluations'][t] for r in rows if r['arm']==arm]) for t in ('0.50','0.75')} for arm in arms}
    return dict(cases=base['cases'],results=rows,summary=summary)


def run(source,out,verify=False):
    if not verify and out.exists():raise FileExistsError('refuse to overwrite '+str(out))
    check_manifest(source)
    base=json.loads((source/'report.json').read_text(encoding='utf-8'))
    for name,expected in base['sources'].items():
        if sha(PROJECT/name)!=expected:raise ValueError('baseline source differs: '+name)
    sources={name:sha(PROJECT/name) for name in SOURCE_NAMES}
    if verify:
        count=check_manifest(out)
        report=json.loads((out/'report.json').read_text(encoding='utf-8'))
        if report['sources']!=sources or report['baseline_sha256']!=sha(source/'report.json') or report['protocol_sha256']!=sha(out/'protocol.md'):
            raise ValueError('source or protocol differs')
        q=json.loads((out/'request.json').read_text(encoding='utf-8'));q.update(output=str(out.resolve()),replay=True)
        worker(q)
        if compute(out,base)!=report['analysis']:raise ValueError('evaluation differs')
        return dict(verified=True,files=count,model_calls_replayed=204,evaluation_groups=192)
    from PIL import Image,__version__ as pillow_version
    if sha(MODEL)!=MODEL_SHA:raise ValueError('model differs')
    runtime=asset_root().resolve();_,lock=validate_assets(runtime)
    out.mkdir(parents=True,exist_ok=False)
    put(out/'protocol.md',(PROJECT/'docs/TINYFORMER_PROTOCOL.md').read_bytes())
    for name in ('sources.json','LICENSE','NOTICE','README.md'):put(out/'model-source'/name,(ROOT/name).read_bytes())
    put(out/'selection.json',(source/'selection.json').read_bytes())
    dump(out/'runtime-lock.json',lock)
    cases=[];preparation=[]
    for c in base['cases']:
        key=c['key'];case={k:c[k] for k in ('key','width','height')};case['plans']={}
        put(out/key/'input.png',(source/key/'input.png').read_bytes())
        with Image.open(source/key/'input.png') as im:
            im.load()
            for grid in (1,4):
                plan=[t['window'] for t in json.loads((source/key/f'grid{grid}/result.json').read_text())['tiles']]
                case['plans'][str(grid)]=plan
                for t in plan:
                    start=perf_counter();blob=prepare(im,t);elapsed=(perf_counter()-start)*1000
                    put(out/key/f'grid{grid}'/f'input-{t["index"]}.gz',gzip.compress(blob,mtime=0))
                    preparation.append(dict(key=key,grid=grid,tile=t['index'],preprocess_ms=elapsed))
        cases.append(case)
    dump(out/'preparation.json',dict(pillow=pillow_version,records=preparation))
    q=dict(cases=cases,model=str(MODEL),runtime=str(runtime),output=str(out.resolve()));dump(out/'request.json',q)
    worker(q);worker(dict(q,replay=True))
    analysis=compute(out,base)
    if sources!={name:sha(PROJECT/name) for name in SOURCE_NAMES}:raise ValueError('source changed during run')
    report=dict(kind='development_tinyformer_comparison',analysis=analysis,sources=sources,baseline_sha256=sha(source/'report.json'),
                model_sha256=MODEL_SHA,protocol_sha256=sha(out/'protocol.md'),score_threshold=.3)
    dump(out/'report.json',report)
    template=(PROJECT/'drone_nav/tinyformer_lab.html').read_text(encoding='utf-8')
    put(out/'demo.html',template.replace('__DATA__',json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return analysis['summary']


if __name__=='__main__':
    p=argparse.ArgumentParser(description='航拍适配模型开发对照；不接飞行控制')
    p.add_argument('--source',type=Path,default=PROJECT/'work/object-eval-01');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.source,a.output,a.verify),ensure_ascii=False,indent=2))
