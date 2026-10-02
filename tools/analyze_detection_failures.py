"""来源：本项目原创。只用既有模型输出做漏检定位和四组后处理对照。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.realvision import sha,dump
from drone_nav.object_metrics import evaluate_frame,aggregate
from drone_nav.detection_failures import diagnose,crosses_seam,changes,geometry,REASONS
from tools.object_probe import check_manifest
from tools.setup_vision import put

ARMS=('baseline','no_core','no_nms','no_core_no_nms')
SOURCE_NAMES=('tools/analyze_detection_failures.py','tools/trace_detection.cjs','drone_nav/detection_failures.py',
              'drone_nav/detection_math.cjs','drone_nav/segmentation_tiles.cjs','drone_nav/object_metrics.py',
              'drone_nav/failure_lab.html')


def trace(source,out,verify=False):
    subprocess.run(['node',str(PROJECT/'tools/trace_detection.cjs')],input=json.dumps(dict(input=str(source.resolve()),output=str(out.resolve()),verify=verify)),
                   text=True,check=True,timeout=180)


def compute(source,out):
    baseline=json.loads((source/'report.json').read_text(encoding='utf-8'))
    results=[]
    for case in baseline['cases']:
        key,w,h=case['key'],case['width'],case['height']
        for grid in (1,4):
            record=json.loads((out/f'{key}-grid{grid}.json').read_text(encoding='utf-8'))
            old=next(r for r in baseline['results'] if r['key']==key and r['grid']==grid)
            variants={}
            for arm in ARMS:
                boxes=record['variants'][arm]
                evaluation={t:evaluate_frame(case['annotations'],boxes,w,h,float(t)) for t in ('0.50','0.75')}
                if arm=='baseline' and evaluation!=old['evaluations']:
                    raise ValueError('baseline metrics differ')
                variants[arm]=dict(evaluations=evaluation,changes={t:{g:changes(old['evaluations'][t][g],evaluation[t][g])
                                                                        for g in ('person','vehicle')} for t in evaluation})
            diagnoses=[]
            by_id={b['audit_id']:b for b in record['candidates']}
            for group in ('person','vehicle'):
                for id_ in old['evaluations']['0.50'][group]['missed_annotations']:
                    target=next(g for g in case['annotations'] if g['id']==id_)
                    d=diagnose(target,record)
                    d['box']=target['box'];d['crosses_seam']=crosses_seam(target['box'],w,h,grid)
                    if d['evidence']:
                        d['evidence']['box']=geometry(by_id[d['evidence']['candidate']])
                        if 'suppression' in d['evidence']:
                            d['evidence']['suppression']['box']=geometry(by_id[d['evidence']['suppression']['kept']])
                    diagnoses.append(d)
            seams={}
            for arm in ('baseline','no_core'):
                seams[arm]={}
                for group in ('person','vehicle'):
                    e=variants[arm]['evaluations']['0.50'][group]
                    matched={m['annotation'] for m in e['matches']}
                    valid=matched|set(e['missed_annotations'])
                    counts={label:dict(targets=0,tp=0,fn=0) for label in ('crosses','not_crosses')}
                    for target in case['annotations']:
                        if target['id'] not in valid:continue
                        label='crosses' if crosses_seam(target['box'],w,h,grid) else 'not_crosses'
                        counts[label]['targets']+=1;counts[label]['tp' if target['id'] in matched else 'fn']+=1
                    seams[arm][group]=counts
            results.append(dict(key=key,grid=grid,variants=variants,diagnoses=diagnoses,seams=seams))
    summary={};reason_counts={};transition_counts={};seam_summary={}
    for grid in (1,4):
        rows=[r for r in results if r['grid']==grid]
        summary[str(grid)]={arm:{t:aggregate([r['variants'][arm]['evaluations'][t] for r in rows]) for t in ('0.50','0.75')} for arm in ARMS}
        reason_counts[str(grid)]={g:dict(Counter(d['reason'] for r in rows for d in r['diagnoses'] if d['group']==g)) for g in ('person','vehicle')}
        transition_counts[str(grid)]={arm:{g:{k:sum(len(r['variants'][arm]['changes']['0.50'][g][k]) for r in rows)
                                              for k in ('recovered','lost','retained')} for g in ('person','vehicle')} for arm in ARMS}
        seam_summary[str(grid)]={arm:{g:{label:{k:sum(r['seams'][arm][g][label][k] for r in rows) for k in ('targets','tp','fn')}
                                           for label in ('crosses','not_crosses')} for g in ('person','vehicle')} for arm in ('baseline','no_core')}
    return dict(cases=[{k:c[k] for k in ('key','width','height')} for c in baseline['cases']],results=results,
                summary=summary,reasons=reason_counts,transitions=transition_counts,seams=seam_summary)


def run(source,out,verify=False):
    if not verify and out.exists():raise FileExistsError('refuse to overwrite '+str(out))
    check_manifest(source)
    base=json.loads((source/'report.json').read_text(encoding='utf-8'))
    for name,digest in base['sources'].items():
        if sha(PROJECT/name)!=digest:raise ValueError('baseline source differs: '+name)
    hashes={name:sha(PROJECT/name) for name in SOURCE_NAMES}
    if verify:
        count=check_manifest(out)
        report=json.loads((out/'report.json').read_text(encoding='utf-8'))
        if report['source_report_sha256']!=sha(source/'report.json') or report['sources']!=hashes:
            raise ValueError('input or source differs')
        if report['protocol_sha256']!=sha(out/'protocol.md'):raise ValueError('protocol differs')
        trace(source,out,True)
        if report['analysis']!=compute(source,out):raise ValueError('analysis replay differs')
        return dict(verified=True,files=count,raw_outputs=204,baseline_frame_arms=24)
    out.mkdir(parents=True,exist_ok=False)
    put(out/'protocol.md',(PROJECT/'docs/DETECTION_FAILURE_PROTOCOL.md').read_bytes())
    trace(source,out)
    analysis=compute(source,out)
    for c in analysis['cases']:put(out/c['key']/'input.png',(source/c['key']/'input.png').read_bytes())
    if hashes!={name:sha(PROJECT/name) for name in SOURCE_NAMES}:raise ValueError('source changed during analysis')
    report=dict(kind='development_detection_failure_analysis',source_archive=str(source.resolve()),source_report_sha256=sha(source/'report.json'),
                sources=hashes,protocol_sha256=sha(out/'protocol.md'),analysis=analysis)
    dump(out/'report.json',report)
    template=(PROJECT/'drone_nav/failure_lab.html').read_text(encoding='utf-8')
    put(out/'demo.html',template.replace('__DATA__',json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return dict(reasons=analysis['reasons'],transitions=analysis['transitions'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description='模型输出重放：归属过滤/去重对照与基线漏检定位')
    p.add_argument('--source',type=Path,default=PROJECT/'work/object-eval-01')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.source,a.output,a.verify),ensure_ascii=False,indent=2))
