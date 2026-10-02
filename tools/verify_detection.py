"""来源：本项目原创。只读核验检测归档，以独立逐行位集重算像素中心覆盖。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parent.parent


def reference_coverage(truth, boxes, width, height, source_width, source_height):
    """直接比较源坐标中的像素中心，不使用评价器的 ceil/切片实现。"""
    if min(width, height, source_width, source_height) <= 0 or len(truth) != width * height:
        raise ValueError('label shape')
    output = {}
    for group, target in [('person', 6), ('vehicle', 3)]:
        selected = [b for b in boxes if b['group'] == group]
        # 每个整数的一位表示该行的一个像素；按位或保证重叠框只计一次。
        spans = [(b, sum(1 << x for x in range(width)
                         if b['x1'] <= (x + .5) * source_width / width < b['x2']))
                 for b in selected]
        actual = area = hit = 0
        for y in range(height):
            cy = (y + .5) * source_height / height
            mask = 0
            for b, span in spans:
                if b['y1'] <= cy < b['y2']:
                    mask |= span
            labels = sum(1 << x for x, t in enumerate(truth[y * width:(y + 1) * width])
                         if t == target)
            actual += labels.bit_count()
            area += mask.bit_count()
            hit += (mask & labels).bit_count()
        output[group] = dict(boxes=len(selected), label_pixels=actual,
                             covered_label_pixels=hit, box_union_pixels=area,
                             label_coverage=hit / actual if actual else None,
                             target_fraction_in_boxes=hit / area if area else None,
                             frame_fraction=area / len(truth))
    return output


def verify(out):
    out = Path(out).resolve()
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        file = (out / name).resolve()
        if not file.is_relative_to(out) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('archive file differs: ' + name)
    report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    for name, expected in report['sources'].items():
        file = (PROJECT / name).resolve()
        if not file.is_relative_to(PROJECT) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('source differs: ' + name)
    if hashlib.sha256((out / 'protocol.md').read_bytes()).hexdigest() != report['protocol_sha256']:
        raise ValueError('protocol differs')
    expected_rows = {(key, grid) for key in ['seq1_000000', 'seq2_000000'] for grid in [1, 4]}
    rows = report['results']
    if len(rows) != 4 or {(r['key'], r['grid']) for r in rows} != expected_rows:
        raise ValueError('development rows differ')
    for r in rows:
        folder = out / r['key']
        meta = json.loads((folder / f"grid{r['grid']}" / 'result.json').read_text(encoding='utf-8'))
        if (meta['width'], meta['height']) != (3840, 2160):
            raise ValueError('source dimensions differ')
        if r['boxes'] != meta['boxes'] or (r['source_width'], r['source_height']) != (meta['width'], meta['height']):
            raise ValueError('report boxes or dimensions differ')
        values = reference_coverage((folder / 'truth.u8').read_bytes(), meta['boxes'],
                                    960, 540, meta['width'], meta['height'])
        if values != r['coverage']:
            raise ValueError('independent coverage differs: ' + r['key'])
    # 明确使用当前归档路径，移动目录后不向旧 request 中的输出位置读写。
    request = json.loads((out / 'request.json').read_text(encoding='utf-8'))
    request.update(output=str(out), replay=True)
    subprocess.run(['node', str(PROJECT / 'tools/probe_detection.cjs')],
                   input=json.dumps(request), text=True, check=True, timeout=120)
    return dict(files=len(manifest), coverage_groups=8, raw_outputs=34, read_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='只读核验 YOLOX-S 开发探针归档')
    parser.add_argument('directory', type=Path)
    print(json.dumps(verify(parser.parse_args().directory), ensure_ascii=False))
