"""来源：本项目原创。透视相机开发实验与可复核输出，不控制无人机。"""

import hashlib
import json
from dataclasses import asdict
from math import ceil, sqrt
from pathlib import Path

from .imaging import DISPLAY_COLORS, png_bytes, png_url
from .pinhole import Intrinsics, Pose, observed_grid, reconstruct
from .raycast import demo_world, render
from .vision_experiment import train_model


def distance(a, b):
    return sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def error_summary(errors):
    ordered = sorted(errors)
    return {"count": len(ordered), "mean_m": sum(ordered)/len(ordered) if ordered else None,
            "p95_m": ordered[ceil(len(ordered)*.95)-1] if ordered else None,
            "max_m": max(ordered) if ordered else None}


def camera_cases():
    return [
        {"key": "overhead", "title": "01 · 垂直俯视", "position": (10, 8, 15), "target": (10, 8, 0),
         "note": "观察地面、屋顶及坡面；高处物体在图像中更大。"},
        {"key": "occluded", "title": "02 · 建筑遮挡", "position": (5, 8, 8), "target": (11, 8, 0),
         "note": "镜头朝向建筑后的检查点，但最近交点是建筑。后方没有观测到的格子保持未知。"},
        {"key": "revealed", "title": "03 · 换视角观察", "position": (12, 4, 8), "target": (11, 8, 0),
         "note": "从侧面重新观察同一检查点。这里是预设独立相机位姿，不是自主绕飞。"},
        {"key": "ramp", "title": "04 · 倾斜地面", "position": (4, 7, 6), "target": (4, 12, .3),
         "note": "坡面不再是固定比例贴图；深度经相机方向转换后还原为世界坐标。"},
        {"key": "dark", "title": "05 · 弱光失败", "position": (12, 4, 8), "target": (11, 8, 0), "light": .25,
         "note": "沿用 v0.2 颜色模型，不重新调参。弱光导致大量未知类别；几何距离仍为模拟输入。"},
        {"key": "dropout", "title": "06 · 深度全部丢失", "position": (12, 4, 8), "target": (11, 8, 0), "dropout": 1.0,
         "note": "即使彩色图看起来正常，没有深度也无法还原三维位置；地图保持全未知。"},
        {"key": "noisy", "title": "07 · 深度含噪声", "position": (12, 4, 8), "target": (11, 8, 0), "noise_std_m": .05,
         "note": "注入标准差 5 厘米的深度噪声，观察位置误差；它不是实测传感器误差。"},
    ]


def run_camera_suite(output: Path) -> dict:
    world, k, model = demo_world(), Intrinsics(), train_model()
    data_dir = output / "camera_data"
    data_dir.mkdir()
    model_blob = json.dumps(model.to_dict(), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    (output/"model.json").write_bytes(model_blob)
    manifest = [{"path": "model.json", "sha256": hashlib.sha256(model_blob).hexdigest()}]
    cases = []
    for index, config in enumerate(camera_cases()):
        pose = Pose.look_at(config["position"], config["target"])
        settings = {"seed": 901, "light": config.get("light", 1.0),
                    "dropout": config.get("dropout", 0.0), "noise_std_m": config.get("noise_std_m", 0.0),
                    "max_range_m": 30.0, "tick": index}
        frame, truth = render(world, pose, k, **settings)
        labels = model.predict_image(frame.rgb)
        points = reconstruct(frame)
        grid = observed_grid(points, labels, width=40, height=32)
        errors = [distance(point, truth.points[i]) for i, point in points]
        wrong_errors = []
        for i, point in points:
            # 错误对照：把沿射线距离误当 Z 深度，独立于当前噪声实验。
            slant_range = distance(truth.points[i], pose.position)
            wrong = pose.unproject(i % k.width, i // k.width, slant_range, k)
            wrong_errors.append(distance(wrong, truth.points[i]))
        mask = [DISPLAY_COLORS[label] if label >= 0 else DISPLAY_COLORS[-1] for label in labels]
        depth_rgb = []
        for z in frame.depth_z_m:
            shade = min(255, int(z/20*255)) if z is not None else 0
            depth_rgb.append((shade, shade, shade))
        images = {"rgb": frame.rgb, "mask": mask, "depth": depth_rgb}
        urls = {}
        for name, pixels in images.items():
            path = f"camera_data/{config['key']}-{name}.png"
            blob = png_bytes(k.width, k.height, pixels)
            (output/path).write_bytes(blob)
            manifest.append({"path": path, "sha256": hashlib.sha256(blob).hexdigest()})
            urls[name] = png_url(k.width, k.height, pixels)
        # 可复核完整帧；None 写成 JSON null，禁止 NaN/Infinity 这种非标准 JSON。
        raw = {"frame": asdict(frame), "predicted_labels": labels,
               "reconstructed_points": points, "evaluation_truth": asdict(truth)}
        path = f"camera_data/{config['key']}.json"
        blob = json.dumps(raw, ensure_ascii=False, allow_nan=False).encode("utf-8")
        (output/path).write_bytes(blob)
        manifest.append({"path": path, "sha256": hashlib.sha256(blob).hexdigest()})
        scene_indices = [i for i, p in enumerate(truth.points) if p is not None]
        metrics = {"pixel_count": k.width*k.height, "visible_surface_pixels": len(scene_indices),
                   "depth_points": len(points), "observed_cells": len(grid["cells"]),
                   "unknown_cells": grid["unknown_cells"],
                   "semantic_accuracy": sum(labels[i] == truth.labels[i] for i in scene_indices)/max(1, len(scene_indices)),
                   "unknown_fraction": sum(labels[i] == -1 for i in scene_indices)/max(1, len(scene_indices)),
                   "reconstruction_error": error_summary(errors),
                   "wrong_range_as_z_error": error_summary(wrong_errors)}
        # 检查同一地面点的投影像素：仅用于解释遮挡，不能据此判定整个格子安全。
        probe = (11.0, 8.0, 0.0)
        projected = pose.project(probe, k)
        probe_info = {"point": probe, "status": "out_of_view", "hit": None}
        if projected is not None:
            u, v, z = projected
            px, py = floor_pixel(u), floor_pixel(v)
            if 0 <= px < k.width and 0 <= py < k.height:
                i = py*k.width+px
                actual = frame.depth_z_m[i]
                probe_info.update(pixel=(px, py), expected_z_m=z, observed_z_m=actual,
                                  hit=truth.objects[i], status="missing_depth" if actual is None else
                                  "occluded" if actual < z-.3 else "depth_consistent" if abs(actual-z) <= .3 else "inconsistent")
        cases.append({**config, "render_settings": settings, "intrinsics": asdict(k), "pose": asdict(pose), "images": urls,
                      "grid": grid, "metrics": metrics, "probe": probe_info})
    report = {"version": "0.3.0", "mode": "camera", "model_sha256": model.digest,
              "seed": 901, "depth_convention": "camera_optical_axis_z_m", "world": asdict(world),
              "limitations": "理想透视与几何遮挡；不是照片级渲染、物理飞行或自主导航。未知格不表示空地，观测到表面也不表示可通行。",
              "cases": cases}
    for name, data in (("camera_report.json", report), ("camera_manifest.json", manifest)):
        (output/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    template = (Path(__file__).parent/"camera_lab.html").read_text(encoding="utf-8")
    (output/"demo.html").write_text(template.replace("__CAMERA_DATA__", json.dumps(
        report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")), encoding="utf-8")
    return report


def floor_pixel(value: float) -> int:
    """像素中心是整数；最近像素的区域为 [i-.5, i+.5)。"""
    from math import floor
    return floor(value+.5)
