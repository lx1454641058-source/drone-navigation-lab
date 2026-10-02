"""来源：本项目原创。训练、独立种子评估及图像导航对照实验。"""

import hashlib
import json
from pathlib import Path
from time import perf_counter

from .classifier import ColorModel
from .image_sensor import ImageSensor
from .imaging import CLASSES, DISPLAY_COLORS, png_bytes
from .mission import LandingSite, Scenario, landing_rejection, run_mission
from .rendering import training_image
from .scenarios import scenarios

TRAIN_SEEDS = tuple(range(100, 112))
TEST_SEEDS = tuple(range(200, 206))
SHIFT_SEEDS = tuple(range(300, 306))


def train_model() -> ColorModel:
    return ColorModel.fit([training_image(seed) for seed in TRAIN_SEEDS])


def image_metrics(model: ColorModel, seeds: tuple[int, ...], light: float | None = None) -> dict:
    confusion = [[0]*6 for _ in CLASSES]
    elapsed = 0.0
    for seed in seeds:
        pixels, truth = training_image(seed, light=light)
        started = perf_counter()
        predicted = model.predict_image(pixels)
        elapsed += perf_counter()-started
        for actual, estimate in zip(truth, predicted):
            confusion[actual][estimate if estimate >= 0 else 5] += 1
    total = sum(map(sum, confusion))
    correct = sum(confusion[i][i] for i in range(5))
    rejected = sum(row[5] for row in confusion)
    per_class = {}
    for i, name in enumerate(CLASSES):
        actual_count = sum(confusion[i])
        predicted_count = sum(row[i] for row in confusion)
        intersection = confusion[i][i]
        per_class[name] = {"recall": intersection/actual_count,
                           "iou": intersection/(actual_count+predicted_count-intersection),
                           "pixels": actual_count}
    return {"seeds": seeds, "image_count": len(seeds), "pixel_count": total,
            "accuracy": correct/total, "unknown_fraction": rejected/total,
            "confusion_rows": list(CLASSES), "confusion_columns": [*CLASSES, "unknown"],
            "confusion": confusion, "per_class": per_class,
            "inference_ms_per_image": round(elapsed*1000/len(seeds), 3),
            "interpretation": "合成色块、相同生成器的开发阶段检查；不是自然图像泛化精度"}


def vision_scenarios() -> list[Scenario]:
    return scenarios() + [
        Scenario("occupied_landing", "08 · 识别人形色块并拒降", "目标区域出现合成人员色块及深度凸起，判断不适合降落。",
                 sites=[LandingSite("被占用取餐点", (26, 10), occupied=True)]),
        Scenario("unknown_landing", "09 · 未见过的颜色拒降", "目标区域颜色不在训练分布内，模型输出未知类别。",
                 sites=[LandingSite("未知表面取餐点", (26, 10), surface="unknown")]),
        Scenario("small_landing", "10 · 空地过小拒降", "目标周边存在低矮障碍，巡航可通过但不能在此降落。",
                 sites=[LandingSite("狭窄取餐点", (26, 10), clear_radius_m=1.1)]),
    ]


def geometric_violations(scenario: Scenario, result: dict) -> int:
    """仅评价器使用世界真值。直接距离检查避免复用规划器的膨胀逻辑。"""
    obstacles = scenario.known_obstacles | scenario.hidden_obstacles | scenario.terrain_blocked
    return sum(any(max(abs(frame["position"][0]-x), abs(frame["position"][1]-y)) <= scenario.margin_cells
                   for x, y in obstacles) for frame in result["trace"])


def unsafe_acceptance(scenario: Scenario, result: dict) -> bool:
    if result["metrics"]["terminal_state"] != "READY_TO_LAND":
        return False
    position = tuple(result["trace"][-1]["position"])
    site = next(site for site in scenario.sites if site.cell == position)
    return landing_rejection(site) is not None


def run_vision_suite(output: Path) -> tuple[list[dict], dict]:
    model = train_model()
    (output/"model.json").write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    dataset = output/"dataset"
    dataset.mkdir()
    manifest = []
    for split, seeds in (("train", TRAIN_SEEDS), ("test", TEST_SEEDS), ("dark_shift", SHIFT_SEEDS)):
        for seed in seeds:
            pixels, labels = training_image(seed, light=0.25 if split == "dark_shift" else None)
            raw = png_bytes(40, 40, pixels)
            (dataset/f"{split}-{seed}.png").write_bytes(raw)
            mask = [DISPLAY_COLORS[label] for label in labels]
            (dataset/f"{split}-{seed}-labels.png").write_bytes(png_bytes(40, 40, mask))
            manifest.append({"split": split, "seed": seed, "rgb": f"dataset/{split}-{seed}.png",
                             "labels": f"dataset/{split}-{seed}-labels.png",
                             "sha256": hashlib.sha256(raw).hexdigest(),
                             "label_sha256": hashlib.sha256(bytes(labels)).hexdigest()})
    (output/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"version": "0.2.0", "model_sha256": model.digest,
              "data_origin": "original_procedural_images", "training_seeds": TRAIN_SEEDS,
              "nominal": image_metrics(model, TEST_SEEDS),
              "dark_shift": image_metrics(model, SHIFT_SEEDS, light=0.25),
              "navigation_comparison": [], "limitations":
              "相同图像生成器的分离种子，不是独立真实分布。原型比较不能证明真实飞行能力或学术创新。"}
    results = []
    for index, scenario in enumerate(vision_scenarios()):
        # 两组使用相同帧生成种子和同一几何规划，只改变末端是否使用语义。
        sensor = ImageSensor(scenario, model, seed=700+index)
        result = run_mission(scenario, sensor)
        result["model_sha256"] = model.digest
        result["metrics"]["geometric_violation_frames"] = geometric_violations(scenario, result)
        results.append(result)
        geometry = run_mission(scenario, ImageSensor(scenario, model, semantic=False, preview=False, seed=700+index))
        report["navigation_comparison"].append({"scenario": scenario.key,
            "geometry_only": geometry["metrics"]["terminal_state"],
            "vision_semantic": result["metrics"]["terminal_state"],
            "geometry_unsafe_acceptance": unsafe_acceptance(scenario, geometry),
            "vision_unsafe_acceptance": unsafe_acceptance(scenario, result),
            "vision_geometric_violation_frames": result["metrics"]["geometric_violation_frames"]})
    (output/"evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return results, report
