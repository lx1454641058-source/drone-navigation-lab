"""来源：本项目原创。十二图独立物体开发评估与只读重放，不是官方 mAP。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from drone_nav.realvision import sha, dump, asset_root, validate_assets
from drone_nav.object_metrics import parse_visdrone, evaluate_frame, aggregate
from tools.setup_vision import put

DATA = Path('D:/DroneNavTools/visdrone-eval/selected-v1')
MODEL = Path('D:/DroneNavTools/yolox-probe/yolox_s.onnx')
MODEL_SHA = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
SOURCES = ('tools/object_probe.py', 'tools/object_worker.cjs', 'tools/setup_object_data.py',
           'drone_nav/object_metrics.py', 'drone_nav/detection_math.cjs',
           'drone_nav/segmentation_tiles.cjs', 'drone_nav/object_lab.html')


def check_manifest(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError('file differs: ' + name)
    return len(manifest)


def compute(out, cases):
    rows, annotated = [], []
    for case in cases:
        key, w, h = case['key'], case['width'], case['height']
        annotations = parse_visdrone((out / key / 'annotations.txt').read_text(encoding='utf-8-sig'), w, h)
        annotated.append(dict(case, annotations=annotations))
        for grid in (1, 4):
            raw = json.loads((out / key / f'grid{grid}/result.json').read_text(encoding='utf-8'))
            rows.append(dict(key=key, grid=grid, boxes=raw['boxes'],
                             evaluations={f'{t:.2f}': evaluate_frame(annotations, raw['boxes'], w, h, t) for t in (.5, .75)},
                             inference_ms=sum(t['inference_ms'] for t in raw['tiles'])))
    summary = {str(grid): {t: aggregate([r['evaluations'][t] for r in rows if r['grid'] == grid])
                          for t in ('0.50', '0.75')} for grid in (1, 4)}
    return annotated, rows, summary


def worker(request):
    subprocess.run(['node', str(PROJECT / 'tools/object_worker.cjs')], input=json.dumps(request),
                   text=True, check=True, timeout=900)


def run(out):
    if out.exists():
        raise FileExistsError('refuse to overwrite ' + str(out))
    check_manifest(DATA)
    if sha(MODEL) != MODEL_SHA:
        raise ValueError('model differs')
    if sha(DATA / 'protocol.md') != sha(PROJECT / 'docs/OBJECT_EVAL_PROTOCOL.md'):
        raise ValueError('data protocol differs')
    runtime = asset_root().resolve()
    _, runtime_lock = validate_assets(runtime)
    sources = {name: sha(PROJECT / name) for name in SOURCES}
    data = json.loads((DATA / 'data.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=False)
    for name in ('selection.json', 'data.json', 'protocol.md'):
        put(out / name, (DATA / name).read_bytes())
    dump(out / 'runtime-lock.json', runtime_lock)
    for case in data['cases']:
        for name in ('original.jpg', 'input.png', 'annotations.txt'):
            put(out / case['key'] / name, (DATA / case['key'] / name).read_bytes())
    request = dict(cases=data['cases'], model=str(MODEL), runtime=str(runtime), output=str(out.resolve()))
    dump(out / 'request.json', request)
    worker(request)
    annotated, rows, summary = compute(out, data['cases'])
    worker(dict(request, replay=True))
    if sources != {name: sha(PROJECT / name) for name in SOURCES}:
        raise ValueError('source changed during experiment')
    report = dict(kind='development_object_evaluation', cases=annotated, results=rows, summary=summary,
                  score_threshold=.3, nms_iou=.45, model_sha256=MODEL_SHA, sources=sources,
                  protocol_sha256=sha(out / 'protocol.md'), replay_passed=True)
    dump(out / 'report.json', report)
    template = (PROJECT / 'drone_nav/object_lab.html').read_text(encoding='utf-8')
    put(out / 'demo.html', template.replace('__DATA__', json.dumps(report, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')).encode('utf-8'))
    dump(out / 'manifest.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()})
    return summary


def verify(out):
    count = check_manifest(out)
    report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    for name, expected in report['sources'].items():
        if name not in SOURCES or sha(PROJECT / name) != expected:
            raise ValueError('source differs: ' + name)
    if set(report['sources']) != set(SOURCES) or sha(out / 'protocol.md') != report['protocol_sha256']:
        raise ValueError('source list or protocol differs')
    request = json.loads((out / 'request.json').read_text(encoding='utf-8'))
    request.update(output=str(out.resolve()), replay=True)
    worker(request)
    annotated, rows, summary = compute(out, request['cases'])
    if (annotated, rows, summary) != (report['cases'], report['results'], report['summary']):
        raise ValueError('evaluation replay differs')
    return dict(files=count, model_outputs=204, frame_arms=24, evaluation_groups=96, verified=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='固定十二图独立物体开发评价 / 已有归档只读核验')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.output) if args.verify else run(args.output), ensure_ascii=False, indent=2))
