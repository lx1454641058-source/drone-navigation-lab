"""来源：本项目原创。模拟观测驱动的决策基线，没有图像模型或飞控连接。"""

from dataclasses import dataclass, field
from math import isfinite

from .planning import Cell, astar, inflate


@dataclass(frozen=True)
class LandingSite:
    name: str
    cell: Cell
    surface: str = "paved"
    slope_deg: float = 0.0
    clear_radius_m: float = 3.0
    occupied: bool = False
    confidence: float = 1.0


@dataclass(frozen=True)
class Observation:
    tick: int
    obstacles: frozenset[Cell]
    confidence: float
    valid: bool = True
    source: str = "synthetic_ground_truth"


@dataclass
class Scenario:
    key: str
    title: str
    description: str
    width: int = 30
    height: int = 20
    start: Cell = (3, 10)
    sites: list[LandingSite] = field(default_factory=list)
    known_obstacles: set[Cell] = field(default_factory=set)
    hidden_obstacles: set[Cell] = field(default_factory=set)
    terrain: dict[Cell, float] = field(default_factory=dict)
    degraded_tick: int | None = None
    cruise_altitude_m: float = 8.0
    min_ground_clearance_m: float = 3.0
    sensor_radius_cells: int = 6
    margin_cells: int = 1

    @property
    def terrain_blocked(self) -> set[Cell]:
        return {cell for cell, elevation in self.terrain.items()
                if self.cruise_altitude_m - elevation < self.min_ground_clearance_m}


class SyntheticSensor:
    """有限半径内读取场景真值，只用于验证接口，不模拟遮挡或视觉误差。"""

    source = "synthetic_ground_truth"
    use_terrain_prior = True
    limitations = "二维离散运动；真值模拟观测；没有摄像头推理、飞控、下降或实际交付"

    def observe(self, scenario: Scenario, position: Cell, tick: int) -> Observation:
        nearby = frozenset(cell for cell in scenario.hidden_obstacles
                            if max(abs(cell[0] - position[0]),
                                   abs(cell[1] - position[1])) <= scenario.sensor_radius_cells)
        confidence = 0.2 if scenario.degraded_tick == tick else 1.0
        return Observation(tick, nearby, confidence)


def landing_rejection(site: LandingSite) -> str | None:
    """数值为可调整的仿真默认值，不是实机安全标准。"""
    values = (site.confidence, site.slope_deg, site.clear_radius_m)
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and isfinite(value) for value in values):
        return "降落观测包含无效数值"
    if not 0 <= site.confidence <= 1 or site.slope_deg < 0 or site.clear_radius_m < 0:
        return "降落观测数值越界"
    if site.confidence < 0.8:
        return "降落观测可信程度不足"
    if type(site.occupied) is not bool or site.occupied:
        return "降落区被占用或占用状态无效"
    if site.surface not in {"paved", "landing_pad"}:
        return "地面类型不在当前允许降落范围"
    if site.slope_deg > 5:
        return "坡度超过仿真阈值"
    if site.clear_radius_m < 2:
        return "空地半径不足"
    return None


def run_mission(scenario: Scenario, sensor: SyntheticSensor | None = None,
                max_steps: int = 250) -> dict:
    if not scenario.sites:
        raise ValueError("at least one landing site is required")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    if scenario.sensor_radius_cells < scenario.margin_cells + 1:
        raise ValueError("sensor range must cover the next step and obstacle margin")
    sensor = sensor or SyntheticSensor()
    position = scenario.start
    terrain_prior = scenario.terrain_blocked if getattr(sensor, "use_terrain_prior", True) else set()
    known = set(scenario.known_obstacles) | terrain_prior
    detected: set[Cell] = set()
    site_index = 0
    path: list[Cell] = []
    trace: list[dict] = []
    moves = 0
    plans = 0
    last_tick = -1
    rejected_sites: list[str] = []
    observation_sources: set[str] = set()

    def record(tick: int, state: str, reason: str) -> None:
        trace.append({"tick": tick, "position": position, "state": state,
                      "reason": reason, "goal": scenario.sites[site_index].cell,
                      "site": scenario.sites[site_index].name,
                      "detected": sorted(detected), "path": list(path)})
        if getattr(sensor, "preview", None) is not None:
            trace[-1]["vision"] = sensor.preview
        if getattr(sensor, "landing", None) is not None:
            trace[-1]["landing"] = dict(sensor.landing)

    for tick in range(max_steps):
        try:
            observation = sensor.observe(scenario, position, tick)
        except (ValueError, TypeError, OverflowError) as error:
            record(tick, "SENSOR_HOLD", f"图像或观测格式无效：{error}")
            break
        # 过期、未来、无效或不可靠观测不得被当成“前方无障碍”。
        if (observation.valid is not True or type(observation.tick) is not int
                or observation.tick != tick or observation.tick <= last_tick
                or type(observation.confidence) not in (int, float) or not isfinite(observation.confidence)
                or not 0.8 <= observation.confidence <= 1.0):
            record(tick, "SENSOR_HOLD", "观测失效或可信程度不足，结束本次运动演示")
            break
        last_tick = observation.tick
        if (not isinstance(observation.obstacles, (set, frozenset))
                or any(not isinstance(cell, tuple) or len(cell) != 2
                       or any(type(v) is not int for v in cell)
                       or not (0 <= cell[0] < scenario.width and 0 <= cell[1] < scenario.height)
                       for cell in observation.obstacles)):
            record(tick, "SENSOR_HOLD", "障碍观测坐标类型或范围无效")
            break
        observation_sources.add(observation.source)
        detected.update(observation.obstacles)
        known.update(observation.obstacles)
        if position in inflate(known, scenario.margin_cells):
            record(tick, "EMERGENCY_STOP", "当前位置已处于障碍缓冲区内，不能继续规划")
            break
        site = scenario.sites[site_index]
        if position == site.cell:
            if hasattr(sensor, "assess_landing"):
                assessment = sensor.assess_landing(site.cell)
                reason = None if assessment.get("accepted") is True else assessment.get("reason", "缺少降落证据")
            else:
                reason = landing_rejection(site)
            if reason is None:
                record(tick, "READY_TO_LAND", "候选降落点通过规则检查；尚未模拟下降和交付")
                break
            rejected_sites.append(site.name)
            record(tick, "LANDING_REJECTED", reason)
            if site_index + 1 >= len(scenario.sites):
                record(tick, "NO_SAFE_SITE", "所有候选降落点均不可用，结束演示并等待人工处理")
                break
            site_index += 1
            path = []
            record(tick, "DIVERT", "选择下一个预设备选点，重新计算可达路线")
            continue

        forbidden = inflate(known, scenario.margin_cells)
        if not path or any(cell in forbidden for cell in path):
            path = astar(scenario.width, scenario.height, position,
                         site.cell, known, scenario.margin_cells)
            plans += 1
            if not path:
                if site_index + 1 < len(scenario.sites):
                    record(tick, "UNREACHABLE_SITE", "当前目标无可行路线，尝试备选点")
                    site_index += 1
                    continue
                record(tick, "NO_PATH", "在已知地图和缓冲约束下找不到可行路线")
                break
            record(tick, "PLAN" if plans == 1 else "REPLAN", "依据当前已知障碍和地形计算四邻接路线")
        position = path[1]
        path = path[1:]
        moves += 1
        record(tick, "FLY", "沿规划路线移动一格；未模拟飞行动力学")
    else:
        record(max_steps, "TIMEOUT", "达到步骤上限，停止演示")

    return {
        "schema_version": 1, "scenario": scenario.key, "title": scenario.title,
        "description": scenario.description,
        "perception_source": getattr(sensor, "source", "injected_test_sensor"),
        "observation_sources": sorted(observation_sources),
        "limitations": getattr(sensor, "limitations", "测试注入观测；未验证真实飞行"),
        "width": scenario.width, "height": scenario.height,
        "start": scenario.start, "sites": [{"name": s.name, "cell": s.cell} for s in scenario.sites],
        "known_obstacles": sorted(scenario.known_obstacles),
        "terrain_blocked": sorted(terrain_prior),
        "margin_cells": scenario.margin_cells,
        "metrics": {"terminal_state": trace[-1]["state"], "grid_moves": moves,
                    "path_length_m": float(moves), "cell_size_m": 1.0,
                    "planning_calls": plans, "rejected_sites": rejected_sites},
        "trace": trace,
    }
