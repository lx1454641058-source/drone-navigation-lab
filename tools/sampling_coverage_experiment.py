"""来源：本项目原创。原 RGB-D 渲染器的采样覆盖矩阵与复核。"""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.sampling_coverage import coverage_scenes, scene_world, camera, depth_candidates, front_projection, summarize
from drone_nav.pinhole import Intrinsics, Pose, PerspectiveFrame
from drone_nav.raycast import render
from drone_nav.imaging import png_bytes
from drone_nav.realvision import sha, dump
from tools.setup_vision import put
from tools.object_probe import check_manifest

PARENT=PROJECT/'work/reobservation-01'
SOURCES=('drone_nav/sampling_coverage.py','tools/sampling_coverage_experiment.py',
         'drone_nav/sampling_coverage_lab.html','tests/test_sampling_coverage.py',
         'drone_nav/pinhole.py','drone_nav/raycast.py','drone_nav/imaging.py','drone_nav/rendering.py')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
canonical=lambda v:json.loads(json.dumps(v,allow_nan=False))


def render_input(scene,width,view):
    k,pose=camera(width,view)
    frame,truth=render(scene_world(scene),pose,k,seed=901,tick=view)
    return canonical(dict(scene=scene,width=width,view=view,frame=asdict(frame),
                          target_mask=[i for i,o in enumerate(truth.objects) if o=='target']))


def evaluate(raw):
    f=raw['frame'];scene=raw['scene'];width=raw['width'];view=raw['view']
    k=Intrinsics(**f['intrinsics']);pose=Pose(**f['pose'])
    frame=PerspectiveFrame(k,pose,f['rgb'],f['depth_z_m'],f['tick']);frame.validate()
    expected_k,expected_pose=camera(width,view)
    if k!=expected_k or canonical(asdict(pose))!=canonical(asdict(expected_pose)) or frame.tick!=view:
        raise ValueError('camera metadata differs from fixed protocol')
    candidates=depth_candidates(frame.depth_z_m)
    if candidates!=raw['target_mask']:raise ValueError('depth candidates differ from independent target truth')
    projection=front_projection(scene,k,pose)
    if projection['front_face_sampling_condition'] and not candidates:
        raise ValueError('front face sampling condition contradicted')
    key=scene['key']+f'-w{width}-v{view}'
    return dict(key=key,scene_key=scene['key'],width=width,height=k.height,view=view,phase=scene['phase'],
                distance_m=scene['distance_m'],size_m=scene['size_m'],target_pixels=len(candidates),
                target_mask=candidates,projection=projection,input='frames/'+key+'.json.gz',image='frames/'+key+'.png')


def run(out,verify=False):
    if not verify and out.exists():raise FileExistsError('refuse overwrite '+str(out))
    check_manifest(PARENT);parent=read(PARENT/'report.json')
    for name,h in parent['sources'].items():
        if sha(PROJECT/name)!=h:raise ValueError('parent source changed: '+name)
    hashes={name:sha(PROJECT/name) for name in SOURCES}
    if verify:
        count=check_manifest(out);r=read(out/'report.json');rows=[];representatives=set();rerendered=0
        if r['sources']!=hashes or r['parent_sha256']!=sha(PARENT/'report.json') or r['protocol_sha256']!=sha(out/'protocol.md'):
            raise ValueError('source or protocol changed')
        expected={s['key']:s for s in coverage_scenes()}
        for row in r['rows']:
            raw=json.loads(gzip.decompress((out/row['input']).read_bytes()))
            if raw['scene']!=expected[row['scene_key']]:raise ValueError('world scene changed')
            calculated=evaluate(raw)
            if calculated!=row:raise ValueError('frame evaluation differs')
            f=raw['frame'];k=f['intrinsics']
            if png_bytes(k['width'],k['height'],f['rgb'])!=(out/row['image']).read_bytes():raise ValueError('RGB preview differs')
            representative=(row['width'],row['view'],bool(row['target_pixels']))
            if representative not in representatives:
                if render_input(raw['scene'],row['width'],row['view'])!=raw:raise ValueError('representative rerender differs')
                representatives.add(representative);rerendered+=1
            rows.append(calculated)
        if len(rows)!=864 or len({r['key'] for r in rows})!=864:raise ValueError('incomplete matrix')
        if summarize(rows)!=r['summary']:raise ValueError('summary differs')
        return dict(verified=True,files=count,frames=len(rows),groups=len(r['summary']),rerendered=rerendered)
    out.mkdir(parents=True,exist_ok=False)
    put(out/'protocol.md',(PROJECT/'docs/SAMPLING_COVERAGE_PROTOCOL.md').read_bytes())
    scenes=coverage_scenes();rows=[];started=perf_counter()
    for index,scene in enumerate(scenes):
        for width in (48,96,192):
            for view in (0,1,2):
                raw=render_input(scene,width,view);row=evaluate(raw);rows.append(row)
                put(out/row['input'],gzip.compress(json.dumps(raw,separators=(',',':'),allow_nan=False).encode('utf-8'),mtime=0))
                f=raw['frame'];put(out/row['image'],png_bytes(width,width*3//4,f['rgb']))
        if (index+1)%8==0:print(json.dumps(dict(progress_scenes=index+1,total_scenes=len(scenes),frames=len(rows),elapsed_s=round(perf_counter()-started,2))),flush=True)
    if hashes!={n:sha(PROJECT/n) for n in SOURCES}:raise ValueError('source changed during run')
    summary=summarize(rows)
    r=dict(kind='point-sampling-coverage',sources=hashes,parent_sha256=sha(PARENT/'report.json'),
           protocol_sha256=sha(out/'protocol.md'),scenes=scenes,rows=rows,summary=summary,
           total_pixels=sum(v['width']*v['height'] for v in rows),elapsed_s=perf_counter()-started,
           flight_authorized=False,clearance_authorized=False)
    dump(out/'report.json',r)
    html=(PROJECT/'drone_nav/sampling_coverage_lab.html').read_text(encoding='utf-8')
    put(out/'demo.html',html.replace('__DATA__',json.dumps(r,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    return dict(scenes=len(scenes),frames=len(rows),groups=len(summary),pixels=r['total_pixels'],elapsed_s=round(r['elapsed_s'],2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.output,a.verify),ensure_ascii=False,indent=2))
