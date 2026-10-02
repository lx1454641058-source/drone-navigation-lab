"""来源：本项目原创。固定阈值的一对一物体评价，不是官方 VisDrone/COCO mAP。"""
import math

GROUPS = {1: 'person', 2: 'person', 4: 'vehicle', 5: 'vehicle', 6: 'vehicle', 9: 'vehicle'}
SIZES = ('tiny', 'small', 'medium', 'large')


def box_valid(box, width, height):
    if len(box) != 4 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in box):
        raise ValueError('nonfinite or malformed box')
    x1, y1, x2, y2 = box
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError('box outside image or empty')


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def iou(a, b):
    cross = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return cross / (area(a) + area(b) - cross)


def ignored_fraction(box, regions):
    """按矩形边界分带求精确并集面积，不把重叠忽略区重复累加。"""
    clips = [(max(box[0], r[0]), max(box[1], r[1]), min(box[2], r[2]), min(box[3], r[3])) for r in regions]
    clips = [r for r in clips if r[0] < r[2] and r[1] < r[3]]
    edges = sorted({x for r in clips for x in (r[0], r[2])})
    total = 0
    for left, right in zip(edges, edges[1:]):
        intervals = sorted((r[1], r[3]) for r in clips if r[0] <= left and r[2] >= right)
        end = -math.inf
        length = 0
        for low, high in intervals:
            length += max(0, high - max(low, end))
            end = max(end, high)
        total += (right - left) * length
    return total / area(box)


def parse_visdrone(text, width, height):
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
        raise ValueError('image shape')
    records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.strip().rstrip(',').split(',')
        if len(fields) != 8:
            raise ValueError(f'annotation line {line_no}: expected eight fields')
        try:
            x, y, w, h, score, category, truncation, occlusion = map(int, fields)
            box = [x, y, x + w, y + h]
            box_valid(box, width, height)
            if score not in (0, 1) or category not in range(12) or truncation not in (0, 1) or occlusion not in (0, 1, 2):
                raise ValueError('annotation flag')
        except ValueError as exc:
            raise ValueError(f'annotation line {line_no}: {exc}') from exc
        records.append(dict(id=line_no, box=box, category=category, group=GROUPS.get(category),
                            ignore=score == 0 or category in (0, 11), truncation=truncation, occlusion=occlusion))
    return records


def size_name(box):
    a = area(box)
    return 'tiny' if a < 256 else 'small' if a < 1024 else 'medium' if a < 9216 else 'large'


def evaluate_frame(annotations, predictions, width, height, threshold=.5):
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('IoU threshold')
    if len({g['id'] for g in annotations}) != len(annotations):
        raise ValueError('duplicate annotation id')
    for g in annotations:
        box_valid(g['box'], width, height)
    for p in predictions:
        box_valid([p[k] for k in ('x1', 'y1', 'x2', 'y2')], width, height)
        if p['group'] not in ('person', 'vehicle') or not math.isfinite(p['score']) or not .3 <= p['score'] <= 1:
            raise ValueError('prediction group or score')
    regions = [g['box'] for g in annotations if g['ignore']]
    output = {}
    for group in ('person', 'vehicle'):
        candidates = [g for g in annotations if not g['ignore'] and g['group'] == group]
        excluded = [g['id'] for g in candidates if ignored_fraction(g['box'], regions) > .5]
        gt = [g for g in candidates if g['id'] not in excluded]
        used, matches, false, duplicate, ignored = set(), [], [], [], []
        ordered = sorted((i for i, p in enumerate(predictions) if p['group'] == group), key=lambda i: (-predictions[i]['score'], i))
        for index in ordered:
            p = predictions[index]
            box = [p[k] for k in ('x1', 'y1', 'x2', 'y2')]
            overlaps = [(iou(box, g['box']), j) for j, g in enumerate(gt)]
            eligible = [(overlap, -j) for overlap, j in overlaps if j not in used and overlap >= threshold]
            if eligible:
                overlap, negative = max(eligible)
                j = -negative
                used.add(j)
                matches.append(dict(prediction=index, annotation=gt[j]['id'], iou=overlap))
            elif any(overlap >= threshold for overlap, _ in overlaps):
                false.append(index)
                duplicate.append(index)
            elif ignored_fraction(box, regions) > .5:
                ignored.append(index)
            else:
                false.append(index)
        missed = [g['id'] for j, g in enumerate(gt) if j not in used]
        tp, fp, fn = len(matches), len(false), len(missed)
        strata = {}
        for field, categories in [('size', SIZES), ('occlusion', (0, 1, 2)), ('truncation', (0, 1))]:
            strata[field] = {}
            for category in categories:
                indices = [j for j, g in enumerate(gt) if (size_name(g['box']) if field == 'size' else g[field]) == category]
                hits = sum(j in used for j in indices)
                strata[field][str(category)] = dict(targets=len(indices), tp=hits, fn=len(indices) - hits,
                                                    recall=hits / len(indices) if indices else None)
        output[group] = dict(tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp) if tp + fp else None,
                             recall=tp / (tp + fn) if tp + fn else None, matches=matches,
                             false_predictions=false, duplicate_predictions=duplicate, missed_annotations=missed,
                             ignored_predictions=ignored, excluded_annotations=excluded, strata=strata)
    return output


def aggregate(frames):
    result = {}
    for group in ('person', 'vehicle'):
        tp, fp, fn = (sum(f[group][k] for f in frames) for k in ('tp', 'fp', 'fn'))
        strata = {}
        for field, categories in [('size', SIZES), ('occlusion', (0, 1, 2)), ('truncation', (0, 1))]:
            strata[field] = {}
            for category in map(str, categories):
                rows = [f[group]['strata'][field][category] for f in frames]
                n = sum(r['targets'] for r in rows)
                hits = sum(r['tp'] for r in rows)
                strata[field][category] = dict(targets=n, tp=hits, fn=n - hits, recall=hits / n if n else None)
        result[group] = dict(tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp) if tp + fp else None,
                             recall=tp / (tp + fn) if tp + fn else None, strata=strata,
                             ignored_predictions=sum(len(f[group]['ignored_predictions']) for f in frames),
                             excluded_annotations=sum(len(f[group]['excluded_annotations']) for f in frames),
                             duplicate_predictions=sum(len(f[group]['duplicate_predictions']) for f in frames))
    return result
