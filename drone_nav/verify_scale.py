"""来源：本项目原创。核验分块协议、源类别、指标和逐块分数重放，保持归档只读。"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .realvision import (PROJECT, CLASSES, CLASS_NAMES, COLORS, UAVID_MAP, ade_mapping,
                         asset_root, validate_assets, metrics, confusion, sha)
from .scale_experiment import ARMS, ARM_NAMES, WORKERS, summarize


def verify(output, root=None, replay=True):
    root = (root or asset_root()).resolve()
    lock, runtime = validate_assets(root)
    load = lambda p: json.loads((output / p).read_text(encoding='utf-8'))
    manifest = load('manifest.json')
    expected = {'assets-lock.json', 'runtime-lock.json', 'protocol.md', 'request.json', 'report.json', 'demo.html', *WORKERS}
    for case in lock['cases']:
        key = case['key']
        expected.update(key + '/' + name for name in ('uavid.u8', 'rgb.u8', 'input.png', 'truth.u8', 'truth.png'))
        for arm, grid in zip(ARMS, (1, 2, 4)):
            expected.update(f'{key}/{arm}/' + name for name in ('ade.u8', 'worker.json', 'prediction.u8', 'prediction.png', 'error.png'))
            expected.update(f'{key}/{arm}/tile-{i}.f32.gz' for i in range(grid * grid))
    if set(manifest) != expected:
        raise ValueError('incomplete or unexpected scale artifacts')
    for rel, digest in manifest.items():
        path = (output / rel).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path) != digest:
            raise ValueError('artifact digest/path differs')
    report = load('report.json')
    if report['version'] != '0.13.0' or report['kind'] != 'scale_development':
        raise ValueError('unexpected scale version')
    if load('assets-lock.json') != lock or load('runtime-lock.json') != runtime:
        raise ValueError('asset provenance differs')
    if (report['classes'] != list(CLASSES) or report['class_names'] != list(CLASS_NAMES) or
            report['colors'] != list(map(list, COLORS)) or report['arms'] != list(ARMS) or report['arm_names'] != list(ARM_NAMES)):
        raise ValueError('class/arm metadata differs')
    mapping = ade_mapping(json.loads((root / 'model/config.json').read_text(encoding='utf-8')))
    if report['ade_mapping'] != list(mapping) or report['uavid_mapping'] != list(UAVID_MAP):
        raise ValueError('mapping differs')
    for name in ('model_repository', 'model_revision', 'data_repository', 'data_revision'):
        if report[name] != lock[name]:
            raise ValueError('model/data version differs')
    if report['model_sha256'] != sha(root / 'model/model.onnx'):
        raise ValueError('model digest differs')
    if report['protocol_sha256'] != sha(output / 'protocol.md') or report['protocol_sha256'] != sha(PROJECT / 'docs/VISION_PROTOCOL_V13.md'):
        raise ValueError('protocol differs')
    if len(report['cases']) != len(lock['cases']):
        raise ValueError('case count differs')
    for case, item in zip(report['cases'], lock['cases']):
        if any(case[k] != v for k, v in item.items()) or set(case['arms']) != set(ARMS):
            raise ValueError('case/split/arms differ')
        directory = output / case['key']
        raw = (directory / 'uavid.u8').read_bytes()
        if any(v >= len(UAVID_MAP) for v in raw):
            raise ValueError('invalid raw label')
        truth = bytes(UAVID_MAP[v] for v in raw)
        if truth != (directory / 'truth.u8').read_bytes():
            raise ValueError('truth mapping differs')
        for arm, grid in zip(ARMS, (1, 2, 4)):
            record = case['arms'][arm]
            worker = load(case['key'] + '/' + arm + '/worker.json')
            if worker != record['worker'] or worker['grid'] != grid or worker['halo'] != .125 or len(worker['tiles']) != grid * grid:
                raise ValueError('worker configuration differs')
            n = worker['width'] * worker['height']
            if (worker['width'] != 960 or worker['height'] != round(worker['source_height'] * 960 / worker['source_width'])
                    or len(truth) != n or (directory / 'rgb.u8').stat().st_size != n * 3):
                raise ValueError('evaluation shape differs')
            ade = (directory / arm / 'ade.u8').read_bytes()
            if len(ade) != n or any(v >= len(mapping) for v in ade):
                raise ValueError('invalid raw prediction')
            pred = bytes(mapping[v] for v in ade)
            if pred != (directory / arm / 'prediction.u8').read_bytes():
                raise ValueError('prediction mapping differs')
            if metrics(confusion(truth, pred)) != record['metrics']:
                raise ValueError('evaluation metrics differ')
            for key in ('preprocess_ms', 'inference_ms', 'postprocess_ms', 'archive_ms'):
                if worker[key] != sum(t[key] for t in worker['tiles']):
                    raise ValueError('timing sum differs')
            if worker['core_ms'] != worker['decode_ms'] + worker['preprocess_ms'] + worker['inference_ms'] + worker['postprocess_ms']:
                raise ValueError('core timing differs')
    if summarize(report['cases']) != report['summary']:
        raise ValueError('aggregate metrics differ')
    if replay:
        node = shutil.which('node')
        if not node:
            raise RuntimeError('Node.js missing')
        for name in WORKERS:
            if report['workers'][name] != sha(output / name) or report['workers'][name] != sha(Path(__file__).parent / name):
                raise ValueError('worker source differs')
        request = dict(action='replay', output=str(output.resolve()), cases=lock['cases'])
        result = subprocess.run([node, str(Path(__file__).with_name('scale_worker.cjs')), '-'],
                                input=json.dumps(request), text=True, capture_output=True, check=True, timeout=900)
        if json.loads(result.stdout) != dict(verified=True, images=9, predictions=27):
            raise ValueError('unexpected replay result')
    return dict(verified=True, images=9, predictions=27, tiles=189, logits_replayed=replay,
                scope='固定分数采样/拼接与指标重算，未重做网络推理，不代表实机能力')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='重放分块尺度实验')
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.root, not args.metrics_only), ensure_ascii=False))
