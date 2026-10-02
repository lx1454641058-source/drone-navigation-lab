"""来源：本项目原创。从保存的类别及分数重算指标；只读实验目录，不宣称重做模型推理。"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .classifier import ColorModel
from .realvision import (PROJECT, CLASSES, CLASS_NAMES, COLORS, COLOR_MAP, UAVID_MAP,
                         ade_mapping, aggregate, confusion, metrics, sha, asset_root, validate_assets)


def read_maps(directory, n, mapping):
    values = {name: (directory / f'{name}.u8').read_bytes() for name in
              ('ade', 'uavid', 'truth', 'segformer', 'color', 'color-original')}
    if any(len(v) != n for v in values.values()):
        raise ValueError('label shape differs')
    try:
        expected = dict(truth=bytes(UAVID_MAP[v] for v in values['uavid']),
                        segformer=bytes(mapping[v] for v in values['ade']),
                        color=bytes(COLOR_MAP[v] for v in values['color-original']))
    except (IndexError, KeyError) as error:
        raise ValueError('invalid source labels') from error
    if any(expected[k] != values[k] for k in expected):
        raise ValueError('saved taxonomy mapping differs')
    return values


def verify(output, root=None, replay_logits=True):
    root = (root or asset_root()).resolve()
    lock, runtime = validate_assets(root)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    top = {'color-model.json', 'assets-lock.json', 'runtime-lock.json', 'protocol.md',
           'request.json', 'report.json', 'demo.html', 'segmentation_worker.cjs'}
    # 根目录八个记录文件；每个样本有十四个产物。
    leaf = {'rgb.u8', 'uavid.u8', 'ade.u8', 'logits.f32.gz', 'input.png', 'worker.json',
            'truth.u8', 'truth.png', 'segformer.u8', 'segformer.png', 'color.u8', 'color.png',
            'color-original.u8', 'error.png'}
    expected = top | {c['key'] + '/' + name for c in lock['cases'] for name in leaf}
    if set(manifest) != expected:
        raise ValueError('incomplete or unexpected manifest')
    for rel, digest in manifest.items():
        path = (output / rel).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path) != digest:
            raise ValueError('artifact hash/path differs')
    load = lambda file: json.loads((output / file).read_text(encoding='utf-8'))
    report = load('report.json')
    if report['version'] != '0.12.0' or report['kind'] != 'realvision_development':
        raise ValueError('unexpected experiment version')
    if load('assets-lock.json') != lock or load('runtime-lock.json') != runtime:
        raise ValueError('saved assets differ')
    if report['protocol_sha256'] != sha(output / 'protocol.md') or (
            report['protocol_sha256'] != sha(PROJECT / 'docs/VISION_PROTOCOL_V12.md')):
        raise ValueError('protocol differs')
    mapping = ade_mapping(json.loads((root / 'model/config.json').read_text(encoding='utf-8')))
    if (report['ade_mapping'] != list(mapping) or report['uavid_mapping'] != list(UAVID_MAP) or
            report['color_mapping'] != {str(k): v for k, v in COLOR_MAP.items()} or
            report['classes'] != list(CLASSES) or report['class_names'] != list(CLASS_NAMES) or
            report['colors'] != list(map(list, COLORS))):
        raise ValueError('mapping metadata differs')
    for name in ('model_repository', 'model_revision', 'data_repository', 'data_revision'):
        if report[name] != lock[name]:
            raise ValueError('provenance differs')
    model = ColorModel.from_dict(load('color-model.json'))
    if report['model_sha256'] != sha(root / 'model/model.onnx') or report['color_model_digest'] != model.digest:
        raise ValueError('model digest differs')
    cases = report['cases']
    if len(cases) != len(lock['cases']):
        raise ValueError('case count differs')
    for item, case in zip(lock['cases'], cases):
        if any(case[k] != v for k, v in item.items()):
            raise ValueError('case/split differs')
        directory = output / case['key']
        worker = load(case['key'] + '/worker.json')
        if worker != case['worker'] or worker['width'] != 960 or worker['height'] != round(
                worker['source_height'] * 960 / worker['source_width']):
            raise ValueError('worker metadata differs')
        n = worker['width'] * worker['height']
        maps = read_maps(directory, n, mapping)
        if len((directory / 'rgb.u8').read_bytes()) != n * 3:
            raise ValueError('RGB shape differs')
        for name in ('segformer', 'color'):
            recomputed = metrics(confusion(maps['truth'], maps[name]))
            if recomputed != case['metrics'][name]:
                raise ValueError('metrics differ: ' + case['key'])
        if case['color_unknown_fraction'] != maps['color-original'].count(255) / n:
            raise ValueError('color rejection differs')
    if aggregate(cases) != report['summary']:
        raise ValueError('aggregate differs')
    if replay_logits:
        worker = Path(__file__).with_name('segmentation_worker.cjs')
        if report['worker_sha256'] != sha(worker) or report['worker_sha256'] != sha(output / 'segmentation_worker.cjs'):
            raise ValueError('logits replay worker changed')
        node = shutil.which('node')
        if not node:
            raise RuntimeError('Node.js missing')
        request = dict(action='replay', output=str(output.resolve()), cases=lock['cases'])
        result = subprocess.run([node, str(worker), '-'], input=json.dumps(request), text=True,
                                capture_output=True, check=True, timeout=600)
        replay = json.loads(result.stdout)
        if replay != dict(replay_verified=True, cases=9):
            raise ValueError('unexpected logits replay')
    return dict(verified=True, cases=len(cases), logits_replayed=replay_logits,
                validation_pixels=report['summary']['validation']['segformer']['pixel_count'],
                scope='保存的分数后处理、类别映射和指标重算；未重做神经网络推理，不验证镜像等同于作者原档')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='核验真实航拍实验归档')
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.root, not args.metrics_only), ensure_ascii=False))
