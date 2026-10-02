"""来源：本项目原创。世界仅供相机渲染和独立轨迹评价；规划模块不读取几何真值。"""

import hashlib
import json
from dataclasses import asdict
from math import sqrt
from pathlib import Path

from .exploration import explore
from .raycast import Box, Surface, World, render


class SimulatedDepthCamera:
    def __init__(self, world: World, dropout_tick: int | None = None, record: bool = False):
        self.world, self.dropout_tick, self.record = world, dropout_tick, record
        self.frames = []

    def capture(self, pose, intrinsics, tick):
        frame, _ = render(self.world, pose, intrinsics, tick=tick, seed=1201,
                          dropout=1.0 if tick == self.dropout_tick else 0.0)
        if self.record:
            self.frames.append(asdict(frame))
        return frame


def exploration_cases():
    # 规划范围外仍有背景地面，避免把地图边界误建模成“世界消失”。
    # 地图外始终禁止移动；相机可看到背景地面，但地图只接收边界内体素。
    ground = Surface("ground", 0, (-40, 60, -40, 56))
    obstacle = Box("building", 3, (8, 6, 0), (10, 10, 6))
    return [
        {"key": "around_building", "title": "01 · 观察后绕开建筑", "world": World((ground, obstacle)),
         "start": (3, 8), "goal": (16, 8), "dropout_tick": None},
        {"key": "blocked_world", "title": "02 · 隔墙目标无法确认路线", "world": World((ground, Box("wall", 3, (8, 0, 0), (10, 16, 6)))),
         "start": (3, 8), "goal": (16, 8), "dropout_tick": None},
        {"key": "depth_failure", "title": "03 · 深度失效立即停止", "world": World((ground, obstacle)),
         "start": (3, 8), "goal": (16, 8), "dropout_tick": 3},
    ]


def segment_box_distance(start, end, box: Box) -> float:
    """独立评价：四邻接水平线段的包围盒即线段本身，轴间隔给出精确最近距离。"""
    if sum(a != b for a, b in zip(start, end)) > 1:
        raise ValueError("audit only supports axis-aligned movements")
    gaps = [max(low-max(a, b), min(a, b)-high, 0)
            for a, b, low, high in zip(start, end, box.low, box.high)]
    return sqrt(sum(gap*gap for gap in gaps))


def audit_motion(world: World, result: dict) -> dict:
    distances = []
    for move in result["moves"]:
        a, b = move["from"], move["to"]
        start, end = (a[0]+.5, a[1]+.5, 3.5), (b[0]+.5, b[1]+.5, 3.5)
        distances.extend(segment_box_distance(start, end, box) for box in world.surfaces if isinstance(box, Box))
    return {"assumed_body_radius_m": .25, "clearance_violations": sum(d <= .25 for d in distances),
            "minimum_center_to_box_m": min(distances) if distances else None,
            "scope": "仅当前长方体静态障碍、四邻接直线段及假设球体半径；不评价真实飞行或降落"}


def run_exploration_suite(output: Path) -> dict:
    results, manifest = [], []
    for config in exploration_cases():
        camera = SimulatedDepthCamera(config["world"], config["dropout_tick"], record=True)
        result = explore(camera, 20, 16, config["start"], config["goal"])
        result.update(key=config["key"], title=config["title"], world=asdict(config["world"]),
                      dropout_tick=config["dropout_tick"], seed=1201,
                      audit=audit_motion(config["world"], result))
        raw_path = config["key"]+"-observations.json"
        blob = json.dumps(camera.frames, allow_nan=False, separators=(",", ":")).encode("utf-8")
        (output/raw_path).write_bytes(blob)
        manifest.append({"path": raw_path, "frames": len(camera.frames),
                         "sha256": hashlib.sha256(blob).hexdigest()})
        results.append(result)
    report = {"version": "0.4.0", "mode": "explore", "results": results}
    (output/"exploration_report.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output/"observations_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    template = (Path(__file__).parent/"exploration_lab.html").read_text(encoding="utf-8")
    (output/"demo.html").write_text(template.replace("__EXPLORATION_DATA__", json.dumps(
        report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")), encoding="utf-8")
    return report
