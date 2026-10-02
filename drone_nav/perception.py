"""来源：本项目原创。只用 RGB-D 像素和相机标定信息感知环境，无场景真值入口。"""

from collections import Counter
from dataclasses import dataclass
from math import atan, degrees, hypot, isfinite, sqrt

from .classifier import ColorModel
from .imaging import CLASSES, DISPLAY_COLORS, RGBDFrame, png_url
from .surface_geometry import fit_plane


@dataclass
class ImageAnalysis:
    labels: list[int]
    obstacles: frozenset[tuple[int, int]]
    valid_fraction: float
    exposed_fraction: float
    unknown_fraction: float
    cell_depths: dict[tuple[int, int], list[float]]


def analyze_frame(frame: RGBDFrame, model: ColorModel, world_size: tuple[int, int],
                  min_clearance_m: float) -> ImageAnalysis:
    frame.validate()
    if not isfinite(min_clearance_m) or min_clearance_m <= 0:
        raise ValueError("invalid minimum clearance")
    labels = model.predict_image(frame.rgb)
    cells: dict[tuple[int, int], list[float]] = {}
    valid = exposed = unknown = total = 0
    for i, (rgb, depth) in enumerate(zip(frame.rgb, frame.depth_m)):
        cell = (frame.origin_cell[0]+(i % frame.width)//frame.pixels_per_cell,
                frame.origin_cell[1]+(i // frame.width)//frame.pixels_per_cell)
        if not (0 <= cell[0] < world_size[0] and 0 <= cell[1] < world_size[1]):
            continue
        total += 1
        unknown += labels[i] == -1
        exposed += sum(rgb) >= 60
        measurements = cells.setdefault(cell, [])
        if type(depth) in (int, float) and isfinite(depth) and 0 < depth <= 30:
            measurements.append(depth)
            valid += 1
    # 距离未知的区域按不可通行处理，不能把缺失深度等同于空地。
    minimum_pixels = frame.pixels_per_cell**2 * 0.8
    obstacles = frozenset(cell for cell, values in cells.items()
                          if len(values) < minimum_pixels or min(values) < min_clearance_m)
    return ImageAnalysis(labels, obstacles, valid/max(total, 1), exposed/max(total, 1),
                         unknown/max(total, 1), cells)


def landing_evidence(frame: RGBDFrame, analysis: ImageAnalysis,
                     goal: tuple[int, int], semantic: bool = True) -> dict:
    """检查半径 2 米内全部像素；坡度由深度点拟合平面得到，不读取预设坡度。"""
    samples: list[tuple[float, float, float, int]] = []
    expected = invalid = 0
    for i, depth in enumerate(frame.depth_m):
        x = frame.origin_cell[0]+((i % frame.width)+0.5)/frame.pixels_per_cell-(goal[0]+0.5)
        y = frame.origin_cell[1]+((i // frame.width)+0.5)/frame.pixels_per_cell-(goal[1]+0.5)
        if hypot(x, y) > 2.0:
            continue
        expected += 1
        if type(depth) not in (int, float) or not isfinite(depth) or not 0 < depth <= 30:
            invalid += 1
        else:
            samples.append((x, y, frame.camera_altitude_m-depth, analysis.labels[i]))
    # 要求整个圆形窗口在当前图像内，局部看见几块地面不足以判定可降落。
    edges = (goal[0]+0.5-frame.origin_cell[0], goal[1]+0.5-frame.origin_cell[1],
             frame.origin_cell[0]+frame.width/frame.pixels_per_cell-(goal[0]+0.5),
             frame.origin_cell[1]+frame.height/frame.pixels_per_cell-(goal[1]+0.5))
    evidence = {"source": "rgbd_plane_fit", "radius_m": 2.0, "slope_deg": None,
                "roughness_m": None, "surface": "unknown", "occupied": False,
                "accepted": False, "reason": "图像覆盖或深度数据不足", "pixel_count": len(samples)}
    if min(edges) < 2.0 or len(samples) < 9 or invalid or not expected:
        return evidence
    try:
        a,b,c = fit_plane((x,y,z) for x,y,z,_ in samples)
    except ValueError:
        evidence["reason"] = "地面点分布不足以拟合平面"
        return evidence
    slope = degrees(atan(hypot(a, b)))
    residual = sqrt(sum((z-(a*x+b*y+c))**2 for x, y, z, _ in samples)/len(samples))
    counts = Counter(label for _, _, _, label in samples)
    dominant = counts.most_common(1)[0][0]
    occupied = any(label in (3, 4) for _, _, _, label in samples)
    evidence.update(slope_deg=round(slope, 3), roughness_m=round(residual, 4),
                    surface=CLASSES[dominant] if dominant >= 0 else "unknown", occupied=occupied)
    if semantic and any(label != 0 for _, _, _, label in samples):
        evidence["reason"] = "候选降落范围含非铺装地面、人员、障碍或未知像素"
    elif slope > 5:
        evidence["reason"] = "深度拟合得到的坡度超过 5 度仿真阈值"
    elif residual > 0.08:
        evidence["reason"] = "地面不平整或存在凸起"
    else:
        evidence.update(accepted=True, reason="图像与深度规则通过；未模拟下降和触地")
    return evidence


def frame_preview(frame: RGBDFrame, analysis: ImageAnalysis) -> dict:
    mask = [DISPLAY_COLORS[label] if label >= 0 else DISPLAY_COLORS[-1] for label in analysis.labels]
    depth_pixels = []
    for value in frame.depth_m:
        shade = max(0, min(255, int(value/12*255))) if isfinite(value) and value > 0 else 0
        depth_pixels.append((shade, shade, shade))
    return {"rgb": png_url(frame.width, frame.height, frame.rgb),
            "mask": png_url(frame.width, frame.height, mask),
            "depth": png_url(frame.width, frame.height, depth_pixels),
            "width": frame.width, "height": frame.height,
            "valid_depth_fraction": round(analysis.valid_fraction, 4),
            "unknown_fraction": round(analysis.unknown_fraction, 4)}
