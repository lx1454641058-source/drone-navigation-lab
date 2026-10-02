"""来源：本项目原创。真实 RGB-D 数据准备、模型调用及只读重放。"""
import argparse
import gzip
import html
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drone_nav.realvision import sha, dump
from drone_nav.semantic_sensor import TinyFormerSession
from drone_nav.tum_rgbd import load_observation, nearest, project_record, rows, safe_path
from tools.tinyformer_probe import prepare as prepare_tensor, ROOT as MODEL_ROOT

PREFIX = 'rgbd_dataset_freiburg3_walking_xyz/'
OFFSETS = (2,4,6,8,10,12)
SOURCE_URL = 'https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz'
OWN = ('drone_nav/tum_rgbd.py', 'tools/tum_rgbd_experiment.py',
       'tests/test_tum_rgbd.py', 'docs/TUM_RGBD_PROTOCOL.md',
       'tools/tum_bag_subset.py', 'tests/test_tum_bag_subset.py')


def manifest(directory):
    return {p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name != 'manifest.json'}


def check_manifest(directory):
    expected = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if not expected or manifest(directory) != expected:
        raise ValueError('archive manifest differs')
    return len(expected)


def prepare_dataset(archive, output):
    # A fixed download size is observable in the official HTTPS response. This
    # detects truncation; the SHA below is local provenance, not upstream attestation.
    if archive.stat().st_size != 527550055:
        raise ValueError('official archive is incomplete or has changed size')
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'rgb.txt', 'depth.txt', 'groundtruth.txt'}
    names = set()
    with tarfile.open(archive, mode='r|gz') as package:
        for member in package:
            if member.name.startswith(PREFIX) and member.name[len(PREFIX):] in metadata:
                relative = member.name[len(PREFIX):]
                if not member.isfile() or member.size > 2_000_000 or relative in names:
                    raise ValueError('invalid or duplicate metadata')
                with safe_path(output,relative).open('xb') as stream:
                    stream.write(package.extractfile(member).read())
                names.add(relative)
    if names != metadata:
        raise ValueError('required original indexes missing')
    rgb, depth, pose = (rows(output/name, width) for name,width in
                       [('rgb.txt',2),('depth.txt',2),('groundtruth.txt',8)])
    origin = min(rgb[0][0], depth[0][0], pose[0][0])
    selected = []
    for offset in OFFSETS:
        target = rgb[0][0]+offset
        rgb_row = nearest(rgb,target)
        try:
            selection = dict(offset_s=offset, requested_rgb_at_s=target,
                rgb=rgb_row, depth=nearest(depth,rgb_row[0]), pose=nearest(pose,rgb_row[0]))
        except ValueError as exc:
            selection = dict(offset_s=offset, requested_rgb_at_s=target,rgb=rgb_row,error=str(exc))
        selected.append(selection)
    wanted = {r[k][1] for r in selected for k in ('rgb','depth') if k in r}
    extracted = set()
    with tarfile.open(archive, mode='r|gz') as package:
        for member in package:
            if not member.name.startswith(PREFIX):
                continue
            relative = member.name[len(PREFIX):]
            if relative not in wanted:
                continue
            if not member.isfile() or member.size > 4_000_000 or relative in extracted:
                raise ValueError('invalid or duplicate selected PNG')
            target = safe_path(output,relative)
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:
                stream.write(package.extractfile(member).read())
            extracted.add(relative)
    if extracted != wanted:
        raise ValueError('selected originals missing')
    dump(output/'selection.json',dict(origin_s=origin,offsets_s=OFFSETS,frames=selected))
    dump(output/'sources.json',dict(url=SOURCE_URL,archive_sha256=sha(archive),bytes=archive.stat().st_size,
        upstream_checksum_available=False,license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
        authors='J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers; TUM RGB-D benchmark',
        publication='A Benchmark for the Evaluation of RGB-D SLAM Systems, IROS 2012',
        license_source='https://cvg.cit.tum.de/data/datasets/rgbd-dataset',
        format_source='https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats',
        modifications='Six fixed timestamp selections; originals unchanged; no third-party code used'))
    dump(output/'manifest.json',manifest(output))
    return dict(prepared=True,files=check_manifest(output),frames=len(selected),
                matched=sum('error' not in r for r in selected))


def sources():
    parent = json.loads((ROOT/'work/semantic-move-01/report.json').read_text(encoding='utf-8'))
    for name, expected in parent['sources'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('frozen baseline differs: '+name)
    return dict(parent['sources'], **{name:sha(ROOT/name) for name in OWN})


def page(report):
    cards = []
    for row in report['frames']:
        if 'error' in row:
            cards.append('<section><h2>'+str(row['offset_s'])+' s</h2><p>'+html.escape(row['error'])+'</p></section>')
            continue
        record = row['call']
        boxes = record['result']['boxes']
        rects = ''.join(f'<rect x="{b["x1"]}" y="{b["y1"]}" width="{b["x2"]-b["x1"]}" height="{b["y2"]-b["y1"]}"/><text x="{b["x1"]}" y="{max(14,b["y1"]-4)}">{b["group"]} {b["score"]:.2f}</text>' for b in boxes)
        samples = [p for o in row['spatial']['projection']['observations'] for p in o['samples']]
        points = ''.join(f'<circle cx="{p["pixel"][0]}" cy="{p["pixel"][1]}" r="2.5"/>' for p in samples)
        detail = dict(rgb_depth_delta_s=row['rgb_depth_delta_s'],rgb_pose_delta_s=row['rgb_pose_delta_s'],
            elapsed_load_and_model_s=row['elapsed_load_and_model_s'],spatial=row['spatial'],timed=row['timed'])
        cards.append(f'''<section><h2>起始后 {row['offset_s']} 秒</h2>
<p>{len(boxes)} 个模型框 · {len(samples)} 个实测深度采样点 · 加载及处理 {row['elapsed_load_and_model_s']:.3f} 秒</p>
<div class="frame"><img alt="TUM 原始彩色图及模型框" src="sensor/{record['directory']}/input.png">
<svg viewBox="0 0 640 480"><g class="boxes">{rects}</g><g class="points">{points}</g></svg></div>
<p>离线空间检查：<b>{row['spatial']['reason']}</b><br>计入本机处理：<b>{row['timed']['reason']}</b></p>
<details><summary>时间、三维点及拒绝原因</summary><pre>{html.escape(json.dumps(detail,ensure_ascii=False,indent=2))}</pre></details></section>''')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实 RGB-D 检测与空间对应</title><style>body{font:16px/1.65 system-ui;background:#eef2f6;color:#182637;max-width:1300px;margin:auto;padding:28px}h1{font-size:30px}header,section{background:white;padding:24px;border-radius:12px;margin-bottom:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}section{min-width:0}.frame{position:relative}.frame img{width:100%;display:block}.frame svg{position:absolute;inset:0;width:100%;height:100%}.boxes rect{stroke:#ffdb37;fill:none;stroke-width:2}.boxes text{fill:#ffdb37;font-size:13px;stroke:#111;paint-order:stroke;stroke-width:2px}.points{fill:#fb445a}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto;font-size:12px}b{overflow-wrap:anywhere}a{color:#1762ad}</style>
<header><h1>真实 RGB-D：检测框对应到实测空间点</h1><p>六个预先固定时刻 · 室内手持相机 · 离线开发检查</p>
<p>黄框为模型输出，红点为框内有有效深度的采样位置；点可能属于背景，不能当成人物中心或完整障碍。没有目标标注，不给出检测准确率。</p>
<p>保留 TUM 动捕坐标系，尚未对齐导航地图；页面不是无人机飞行记录。空间检查的理想处理时刻与计入本机处理耗时的结果分开显示；两者均不授权飞行。</p>
<p>数据：J. Sturm 等，TUM RGB-D benchmark，IROS 2012，<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>。原图未修改，叠加层由本项目生成。<a href="https://cvg.cit.tum.de/data/datasets/rgbd-dataset">官方来源</a></p></header><main class="grid">'''+''.join(cards)+'</main></html>'


def independent_projection(observation, result, pose_record):
    # Independent quaternion vector formula, without Pose or a matrix library.
    tx,ty,tz,x,y,z,w = map(float,pose_record[1:])
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    x,y,z,w = (a/norm for a in (x,y,z,w))
    def cross(a,b):
        return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def rotate(vector, inverse=False):
        q = (-x,-y,-z) if inverse else (x,y,z)
        first = cross(q,vector); second = cross(q,first)
        return tuple(vector[i]+2*w*first[i]+2*second[i] for i in range(3))
    k = observation.frame.intrinsics
    count = 0
    for item in result['projection']['observations']:
        for p in item['samples']:
            u,v = p['pixel']; d = observation.frame.depth_z_m[v*640+u]
            camera = ((u-k.cx)*d/k.fx,(v-k.cy)*d/k.fy,d)
            expected = tuple(a+b for a,b in zip(rotate(camera),(tx,ty,tz)))
            if max(abs(a-b) for a,b in zip(expected,p['world_m'])) > 1e-10:
                raise ValueError('independent world coordinates differ')
            reconstructed = rotate(tuple(a-b for a,b in zip(p['world_m'],(tx,ty,tz))),inverse=True)
            uv = (k.fx*reconstructed[0]/reconstructed[2]+k.cx,
                  k.fy*reconstructed[1]/reconstructed[2]+k.cy)
            if max(abs(uv[0]-u),abs(uv[1]-v))>1e-8:
                raise ValueError('pixel roundtrip differs')
            count += 1
    return count


def run(dataset, output, verify=False):
    from PIL import Image
    check_manifest(dataset)
    selection = json.loads((dataset/'selection.json').read_text(encoding='utf-8'))
    if selection.get('source_format')=='official ROS bag subset':
        from tools.tum_bag_subset import verify_dataset
        print('verified source subset',verify_dataset(dataset),flush=True)
    if selection['offsets_s'] != list(OFFSETS):
        raise ValueError('fixed timestamp selection differs')
    source_hashes = sources()
    if verify:
        files = check_manifest(output)
        report = json.loads((output/'report.json').read_text(encoding='utf-8'))
        if report['sources'] != source_hashes or report['dataset_manifest_sha256'] != sha(dataset/'manifest.json'):
            raise ValueError('input or source differs')
        if len(report['frames']) != len(selection['frames']):
            raise ValueError('frame count differs')
        sample_count = 0
        with TinyFormerSession(output/'sensor', replay=True) as detector:
            for selected,row in zip(selection['frames'],report['frames']):
                if row['offset_s'] != selected['offset_s']:
                    raise ValueError('offset differs')
                if 'error' in selected:
                    if row != selected: raise ValueError('pairing failure differs')
                    continue
                observation = load_observation(dataset,selected,selection['origin_s'])
                record = row['call']
                if detector.replay_frame(record)['boxes'] != record['result']['boxes']:
                    raise ValueError('model replay differs')
                directory = output/'sensor'/record['directory']
                with Image.open(directory/'input.png') as image:
                    if tuple(image.getdata()) != observation.frame.rgb:
                        raise ValueError('model RGB differs from original')
                    if prepare_tensor(image,record['window']) != gzip.decompress((directory/'input.gz').read_bytes()):
                        raise ValueError('model tensor differs')
                ready = max(observation.captured_at_s,observation.depth_at_s,observation.pose_at_s)
                spatial = project_record(observation,record,completed_at_s=ready,now_s=ready)
                completed = ready+row['elapsed_load_and_model_s']
                timed = project_record(observation,record,completed_at_s=completed,now_s=completed)
                if (spatial,timed) != (row['spatial'],row['timed']):
                    raise ValueError('spatial or consumption replay differs')
                sample_count += independent_projection(observation,spatial,selected['pose'])
                print('verified offset',row['offset_s'],flush=True)
        if page(report) != (output/'demo.html').read_text(encoding='utf-8'):
            raise ValueError('page differs')
        return dict(verified=True,files=files,frames=len(report['frames']),independently_projected_samples=sample_count)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(ROOT/'docs/TUM_RGBD_PROTOCOL.md',output/'protocol.md')
    shutil.copyfile(dataset/'sources.json',output/'dataset-sources.json')
    (output/'model-source').mkdir()
    for name in ('LICENSE','NOTICE','sources.json'):
        shutil.copyfile(MODEL_ROOT/name,output/'model-source'/name)
    report = dict(kind='tum-rgbd-spatial-probe',dataset=str(dataset.resolve()),
        dataset_manifest_sha256=sha(dataset/'manifest.json'),sources=source_hashes,frames=[])
    with TinyFormerSession(output/'sensor') as detector:
        report['model_startup_s'] = detector.startup_s
        for selected in selection['frames']:
            if 'error' in selected:
                report['frames'].append(selected); continue
            start = perf_counter()
            observation = load_observation(dataset,selected,selection['origin_s'])
            record = detector.detect(observation.frame)
            elapsed = perf_counter()-start
            ready = max(observation.captured_at_s,observation.depth_at_s,observation.pose_at_s)
            row = dict(offset_s=selected['offset_s'],call=record,elapsed_load_and_model_s=elapsed,
                rgb_depth_delta_s=observation.depth_at_s-observation.captured_at_s,
                rgb_pose_delta_s=observation.pose_at_s-observation.captured_at_s,
                spatial=project_record(observation,record,completed_at_s=ready,now_s=ready),
                timed=project_record(observation,record,completed_at_s=ready+elapsed,now_s=ready+elapsed))
            report['frames'].append(row)
            print('processed',selected['offset_s'],'boxes',len(record['result']['boxes']),
                  'elapsed',round(elapsed,3),'decision',row['timed']['reason'],flush=True)
    dump(output/'report.json',report)
    (output/'demo.html').write_text(page(report),encoding='utf-8')
    dump(output/'manifest.json',manifest(output))
    return dict(completed=True,frames=len(report['frames']),files=check_manifest(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','run','verify'))
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--dataset',type=Path,default=Path('work/tum-rgbd-input-01'))
    parser.add_argument('--output',type=Path,default=Path('work/tum-rgbd-01'))
    args = parser.parse_args()
    if args.action == 'prepare':
        if not args.archive: parser.error('prepare requires --archive')
        result = prepare_dataset(args.archive,args.dataset)
    else:
        result = run(args.dataset,args.output,verify=args.action=='verify')
    print(json.dumps(result,indent=2),flush=True)
