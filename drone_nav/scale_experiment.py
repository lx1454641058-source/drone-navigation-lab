"""来源：本项目原创。固定分块尺度对照；复用真实视觉指标和资源，不接飞行动作。"""

import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess

from .imaging import png_bytes
from .realvision import (PROJECT, CLASSES, CLASS_NAMES, COLORS, UAVID_MAP, ade_mapping,
                         asset_root, validate_assets, confusion, metrics, dump, sha)

ARMS = ('whole', 'grid2', 'grid4')
ARM_NAMES = ('整幅图像', '2×2 分块', '4×4 分块')
WORKERS = ('segmentation_worker.cjs', 'segmentation_tiles.cjs', 'scale_worker.cjs')


def summarize(cases):
    result = {}
    for split in ('development', 'validation'):
        group = [c for c in cases if c['split'] == split]
        result[split] = {}
        for arm in ARMS:
            matrix = [[sum(c['arms'][arm]['metrics']['confusion'][i][j] for c in group)
                       for j in range(7)] for i in range(7)]
            result[split][arm] = metrics(matrix)
            result[split][arm]['image_count'] = len(group)
            result[split][arm]['median_core_ms'] = statistics.median(c['arms'][arm]['worker']['core_ms'] for c in group)
    return result


def run_suite(output):
    if output.exists():
        raise FileExistsError('refuse to overwrite: ' + str(output))
    root = asset_root().resolve()
    lock, runtime = validate_assets(root)
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node.js missing')
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'assets-lock.json', lock)
    dump(output / 'runtime-lock.json', runtime)
    protocol = PROJECT / 'docs/VISION_PROTOCOL_V13.md'
    (output / 'protocol.md').write_bytes(protocol.read_bytes())
    for name in WORKERS:
        (output / name).write_bytes((Path(__file__).parent / name).read_bytes())
    dump(output / 'request.json', dict(root=str(root), output=str(output.resolve()), cases=lock['cases']))
    subprocess.run([node, str(output / 'scale_worker.cjs'), str(output / 'request.json')], check=True, timeout=1800)
    mapping = ade_mapping(json.loads((root / 'model/config.json').read_text(encoding='utf-8')))
    cases = []
    for item in lock['cases']:
        case = dict(item, arms={})
        directory = output / item['key']
        raw = (directory / 'uavid.u8').read_bytes()
        truth = bytes(UAVID_MAP[v] for v in raw)
        (directory / 'truth.u8').write_bytes(truth)
        rgb = (directory / 'rgb.u8').read_bytes()
        pixels = list(zip(rgb[::3], rgb[1::3], rgb[2::3]))
        for arm in ARMS:
            sub = directory / arm
            worker = json.loads((sub / 'worker.json').read_text(encoding='utf-8'))
            width, height = worker['width'], worker['height']
            ade = (sub / 'ade.u8').read_bytes()
            if len(ade) != width * height or len(truth) != len(ade) or len(rgb) != len(ade) * 3:
                raise ValueError('image shape differs')
            pred = bytes(mapping[v] for v in ade)
            (sub / 'prediction.u8').write_bytes(pred)
            (sub / 'prediction.png').write_bytes(png_bytes(width, height, [COLORS[v] for v in pred]))
            error = [(241, 77, 158) if actual == 6 and estimate != 6 else
                     (236, 129, 75) if actual != estimate else tuple(int(c * .32) for c in pixel)
                     for actual, estimate, pixel in zip(truth, pred, pixels)]
            (sub / 'error.png').write_bytes(png_bytes(width, height, error))
            case['arms'][arm] = dict(worker=worker, metrics=metrics(confusion(truth, pred)))
        (directory / 'truth.png').write_bytes(png_bytes(width, height, [COLORS[v] for v in truth]))
        cases.append(case)
    report = dict(version='0.13.0', kind='scale_development', classes=CLASSES, class_names=CLASS_NAMES,
                  colors=COLORS, arms=ARMS, arm_names=ARM_NAMES, cases=cases, summary=summarize(cases),
                  ade_mapping=list(mapping), uavid_mapping=list(UAVID_MAP), model_sha256=sha(root / 'model/model.onnx'),
                  model_repository=lock['model_repository'], model_revision=lock['model_revision'],
                  data_repository=lock['data_repository'], data_revision=lock['data_revision'],
                  protocol_sha256=sha(protocol), workers={name: sha(output / name) for name in WORKERS},
                  limitations=['原验证图在 v0.12 已观察过，不是全新的隐藏测试集',
                               '评价分辨率低于原始 4K，单帧不提供距离、运动或降落许可',
                               '上下文重叠但只拼接核心，仍可能出现边界不连续',
                               '耗时受机器负载、固定运行顺序和缓存影响；只作本机开发测量'])
    dump(output / 'report.json', report)
    # 复用原视觉实验页面的样式，避免建立另一套界面主题。
    old_template = (Path(__file__).parent / 'realvision_lab.html').read_text(encoding='utf-8')
    css = re.search(r'<style>(.*?)</style>', old_template, re.S).group(1)
    template = (Path(__file__).parent / 'scale_lab.html').read_text(encoding='utf-8')
    html = template.replace('__LAB_CSS__', css).replace('__REPORT_JSON__',
        json.dumps(report, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c'))
    (output / 'demo.html').write_text(html, encoding='utf-8')
    dump(output / 'manifest.json', {str(p.relative_to(output)).replace('\\', '/'): sha(p)
                                    for p in output.rglob('*') if p.is_file()})
    return report
