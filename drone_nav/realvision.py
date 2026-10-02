"""来源：本项目原创。固定自然图像开发评估，标签仅用于评价，不连接飞行动作。"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

from .imaging import png_bytes
from .vision_experiment import train_model

CLASSES = ('other', 'building', 'road', 'vehicle', 'tree', 'vegetation', 'person')
CLASS_NAMES = ('其他', '建筑', '道路', '车辆', '树木', '低矮植被', '行人')
COLORS = ((82, 89, 112), (214, 109, 82), (157, 167, 182), (244, 192, 77),
          (44, 137, 94), (140, 181, 66), (241, 77, 158))
UAVID_MAP = bytes((0, 1, 2, 3, 4, 5, 6, 3))
COLOR_MAP = {0: 2, 1: 5, 2: 0, 3: 1, 4: 6, 255: 0}
ADE_GROUPS = {
    1: ('building', 'house', 'skyscraper', 'hovel', 'tower'),
    2: ('road',), 3: ('car', 'bus', 'truck', 'van'),
    4: ('tree', 'palm'), 5: ('grass', 'plant', 'flower', 'field'), 6: ('person',),
}
PROJECT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def ade_mapping(config):
    labels = config['id2label']
    if set(labels) != {str(i) for i in range(150)}:
        raise ValueError('expected 150 ADE20K classes')
    wanted = {name: group for group, names in ADE_GROUPS.items() for name in names}
    if not set(wanted).issubset(set(labels.values())):
        raise ValueError('model taxonomy differs')
    return bytes(wanted.get(labels[str(i)], 0) for i in range(150))


def confusion(truth, pred):
    if len(truth) != len(pred) or not truth:
        raise ValueError('empty or mismatched label maps')
    matrix = [[0] * 7 for _ in range(7)]
    for actual, estimate in zip(truth, pred):
        if type(actual) is not int or type(estimate) is not int or not 0 <= actual < 7 or not 0 <= estimate < 7:
            raise ValueError('invalid evaluation label')
        matrix[actual][estimate] += 1
    return matrix


def metrics(matrix):
    if len(matrix) != 7 or any(len(row) != 7 for row in matrix) or any(
            type(v) is not int or v < 0 for row in matrix for v in row):
        raise ValueError('invalid confusion matrix')
    total = sum(map(sum, matrix))
    if not total:
        raise ValueError('empty evaluation')
    rows = list(map(sum, matrix))
    cols = [sum(row[i] for row in matrix) for i in range(7)]
    per_class = []
    for i in range(7):
        hit = matrix[i][i]
        union = rows[i] + cols[i] - hit
        per_class.append(dict(name=CLASSES[i], pixels=rows[i], predicted_pixels=cols[i],
                              iou=hit / union if union else None,
                              recall=hit / rows[i] if rows[i] else None))
    ious = [v['iou'] for v in per_class if v['iou'] is not None]
    false_road = cols[2] - matrix[2][2]
    return dict(confusion=matrix, pixel_count=total, per_class=per_class,
                accuracy=sum(matrix[i][i] for i in range(7)) / total,
                mean_iou=sum(ious) / len(ious), iou_class_count=len(ious),
                person_recall=per_class[6]['recall'], false_road_pixels=false_road,
                nonroad_as_road_fraction=false_road / (total - rows[2]) if total != rows[2] else None)


def aggregate(cases):
    result = {}
    for split in ('development', 'validation'):
        result[split] = {}
        selected = [case for case in cases if case['split'] == split]
        for model in ('segformer', 'color'):
            matrix = [[sum(c['metrics'][model]['confusion'][i][j] for c in selected)
                       for j in range(7)] for i in range(7)]
            result[split][model] = metrics(matrix)
            result[split][model]['image_count'] = len(selected)
    return result


def asset_root():
    return Path(os.environ.get('DRONE_VISION_ROOT', 'D:/DroneNavTools/vision-v12'))


def validate_assets(root):
    lock = json.loads((PROJECT / 'tools/vision_assets.json').read_text(encoding='utf-8'))
    for asset in lock['assets']:
        target = root / asset['path']
        if not target.is_file() or target.stat().st_size != asset['bytes'] or sha(target) != asset['sha256']:
            raise ValueError(f"missing or changed vision asset: {target}; run tools/setup_vision.py")
    if json.loads((root / 'assets-lock.json').read_text(encoding='utf-8')) != lock:
        raise ValueError('asset lock differs')
    runtime = json.loads((root / 'runtime-lock.json').read_text(encoding='utf-8'))
    for rel, expected in runtime.items():
        p = root / rel
        if not p.resolve().is_relative_to(root.resolve()) or sha(p) != expected:
            raise ValueError('runtime changed')
    return lock, runtime


def run_suite(output):
    if output.exists():
        raise FileExistsError(f'refuse to overwrite: {output}')
    root = asset_root().resolve()
    lock, runtime = validate_assets(root)
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node.js executable missing')
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((root / 'model/config.json').read_text(encoding='utf-8'))
    mapping = ade_mapping(config)
    model = train_model()
    dump(output / 'color-model.json', model.to_dict())
    dump(output / 'assets-lock.json', lock)
    dump(output / 'runtime-lock.json', runtime)
    protocol = PROJECT / 'docs/VISION_PROTOCOL_V12.md'
    (output / 'protocol.md').write_bytes(protocol.read_bytes())
    dump(output / 'request.json', dict(root=str(root), output=str(output.resolve()), cases=lock['cases']))
    worker = Path(__file__).with_name('segmentation_worker.cjs')
    # 记录并执行本次源码快照，避免运行途中编辑源码导致“记录版本”不等于“执行版本”。
    archived_worker = output / 'segmentation_worker.cjs'
    archived_worker.write_bytes(worker.read_bytes())
    subprocess.run([node, str(archived_worker), str(output / 'request.json')], check=True, timeout=900)
    cases = []
    for item in lock['cases']:
        case = dict(item)
        directory = output / item['key']
        worker_result = json.loads((directory / 'worker.json').read_text(encoding='utf-8'))
        width, height = worker_result['width'], worker_result['height']
        raw_ade = (directory / 'ade.u8').read_bytes()
        raw_truth = (directory / 'uavid.u8').read_bytes()
        rgb = (directory / 'rgb.u8').read_bytes()
        n = width * height
        if len(raw_ade) != n or len(raw_truth) != n or len(rgb) != n * 3:
            raise ValueError('worker output shape mismatch')
        truth = bytes(UAVID_MAP[v] for v in raw_truth)
        pred = bytes(mapping[v] for v in raw_ade)
        pixels = list(zip(rgb[::3], rgb[1::3], rgb[2::3]))
        started = perf_counter()
        raw_color = bytes(p if p >= 0 else 255 for p in model.predict_image(pixels))
        color_ms = (perf_counter() - started) * 1000
        baseline = bytes(COLOR_MAP[p] for p in raw_color)
        for name, values in [('truth', truth), ('segformer', pred), ('color', baseline)]:
            (directory / f'{name}.u8').write_bytes(values)
            (directory / f'{name}.png').write_bytes(png_bytes(width, height, [COLORS[v] for v in values]))
        (directory / 'color-original.u8').write_bytes(raw_color)
        # 错误图只表示类别不一致，粉红色特别标出漏检的标注行人。
        error = [(241, 77, 158) if actual == 6 and estimate != 6 else
                 (236, 129, 75) if actual != estimate else
                 tuple(int(c * .32) for c in pixel)
                 for actual, estimate, pixel in zip(truth, pred, pixels)]
        (directory / 'error.png').write_bytes(png_bytes(width, height, error))
        case.update(worker=worker_result, metrics=dict(segformer=metrics(confusion(truth, pred)),
                                                       color=metrics(confusion(truth, baseline))),
                    color_ms=color_ms, color_unknown_fraction=raw_color.count(255) / n)
        cases.append(case)
        print(item['key'], 'mIoU', case['metrics']['segformer']['mean_iou'], flush=True)
    report = dict(version='0.12.0', kind='realvision_development', classes=CLASSES,
                  class_names=CLASS_NAMES, colors=COLORS, cases=cases, summary=aggregate(cases),
                  model_repository=lock['model_repository'], model_revision=lock['model_revision'],
                  model_sha256=sha(root / 'model/model.onnx'), data_repository=lock['data_repository'],
                  data_revision=lock['data_revision'], ade_mapping=list(mapping),
                  uavid_mapping=list(UAVID_MAP), color_mapping=COLOR_MAP,
                  color_model_digest=model.digest, protocol_sha256=sha(protocol),
                  worker_sha256=sha(archived_worker), runtime_root=str(root),
                  limitations=['九张固定开发样本，验证仅七张，不是官方或正式论文成绩',
                               '镜像标注经过索引格式转换，未与作者原档逐文件复核',
                               '图像缩小至 512×512，评价宽 960，小目标可能丢失',
                               '无距离、动态信息或飞行指令；道路类别不代表允许降落',
                               '预处理尚未与上游实现逐位对齐，类别映射存在差异'])
    dump(output / 'report.json', report)
    template = Path(__file__).with_name('realvision_lab.html').read_text(encoding='utf-8')
    (output / 'demo.html').write_text(template.replace('__REPORT_JSON__', json.dumps(
        report, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')), encoding='utf-8')
    dump(output / 'manifest.json', {str(p.relative_to(output)).replace('\\', '/'): sha(p)
                                    for p in output.rglob('*') if p.is_file()})
    return report
