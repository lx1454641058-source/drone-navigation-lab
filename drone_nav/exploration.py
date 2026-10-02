"""来源：本项目原创。深度体素观测驱动的固定高度探索，不读取世界几何真值。"""

from collections import Counter
from dataclasses import asdict
from math import atan2, cos, pi, sin
from typing import Protocol

from .imaging import png_url
from .occupancy import VoxelMap
from .pinhole import Intrinsics, PerspectiveFrame, Pose, Vec3
from .planning import astar, inflate


class DepthCamera(Protocol):
    def capture(self, pose: Pose, intrinsics: Intrinsics, tick: int) -> PerspectiveFrame: ...


def scan_poses(position: Vec3, target: tuple[int, int]) -> list[Pose]:
    heading = atan2(target[1]+.5-position[1], target[0]+.5-position[0])
    return [Pose.look_at(position, (position[0]+8*cos(heading+i*pi/4),
                                    position[1]+8*sin(heading+i*pi/4), position[2]-3)) for i in range(8)]


def choose_route(grid: VoxelMap, start: tuple[int, int], goal: tuple[int, int],
                 tick: int, visits: Counter, layer: int = 3, pending=None):
    blocked = grid.blocked(layer, tick)
    direct = astar(grid.width, grid.height, start, goal, blocked, margin=1)
    if direct:
        return direct, "GOAL_PATH"
    # 已选的观察位置仍可达时先走完，防止每帧重新选边界造成往返摆动。
    if pending is not None and pending not in (start, goal):
        continuation = astar(grid.width, grid.height, start, pending, blocked, margin=1)
        if continuation:
            return continuation, "FRONTIER_PATH"
    forbidden = inflate(blocked, 1)
    candidates = []
    for y in range(1, grid.height-1):
        for x in range(1, grid.width-1):
            cell = (x, y)
            if cell == start or cell in forbidden or visits[cell] >= 3:
                continue
            # 可达空闲区边缘：两格之外还有未知；隔着障碍不推断后方为空闲。
            boundary = any(grid.state((x+dx, y+dy, layer), tick) == "unknown"
                           for dx in range(-2, 3) for dy in range(-2, 3) if max(abs(dx), abs(dy)) == 2)
            if boundary:
                score = abs(x-goal[0])+abs(y-goal[1])+visits[cell]*4
                candidates.append((score, cell))
    for _, cell in sorted(candidates):
        path = astar(grid.width, grid.height, start, cell, blocked, margin=1)
        if path:
            return path, "FRONTIER_PATH"
    return [], "NO_OBSERVED_ROUTE"


def preview(frame: PerspectiveFrame) -> dict:
    depth = []
    for z in frame.depth_z_m:
        shade = min(255, int(z/20*255)) if z is not None else 0
        depth.append((shade, shade, shade))
    k = frame.intrinsics
    return {"rgb": png_url(k.width, k.height, frame.rgb), "depth": png_url(k.width, k.height, depth),
            "camera_position": frame.pose.position, "camera_forward": frame.pose.forward}


def explore(camera: DepthCamera, width: int, height: int, start: tuple[int, int],
            goal: tuple[int, int], *, max_ticks: int = 70, previews: bool = True) -> dict:
    """先扫描再规划再走一格；起始也必须被观测支持，不植入真值空闲区域。"""
    for cell in (start, goal):
        if (not isinstance(cell, tuple) or len(cell) != 2 or any(type(v) is not int for v in cell)
                or not 1 <= cell[0] < width-1 or not 1 <= cell[1] < height-1):
            raise ValueError("start and goal must be interior integer cells")
    if type(max_ticks) is not int or max_ticks < 1:
        raise ValueError("positive step budget required")
    grid = VoxelMap(width, height)
    k = Intrinsics(40, 30, 25, 25, 19.5, 14.5)
    position, aim = start, goal
    visits: Counter = Counter({start: 1})
    trace, moves = [], []
    for tick in range(max_ticks):
        xyz = (position[0]+.5, position[1]+.5, 3.5)
        frames = []
        state, reason, path, info = "OBSERVE", "", [], {}
        try:
            frames = [camera.capture(pose, k, tick) for pose in scan_poses(xyz, aim)]
            info = grid.integrate(frames, tick, xyz)
            if info["valid_depth_fraction"] < .05:
                state, reason = "SENSOR_HOLD", "本轮有效深度不足 5%，不使用旧地图继续移动"
            else:
                path, route_kind = choose_route(grid, position, goal, tick, visits, pending=aim)
                if not path:
                    state, reason = "NO_OBSERVED_ROUTE", "没有找到由近期观测支持的路线或可探索边界；不能据此断言全局无路"
                elif position == goal:
                    state, reason = "ARRIVED_WAYPOINT", "抵达固定高度的目标航点；未下降、未交付"
                else:
                    nxt = path[1]
                    # 再核对下一格及其 1 格水平缓冲，避免使用失效路径。
                    if any(grid.state((nxt[0]+dx, nxt[1]+dy, 3), tick) != "free"
                           for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                        state, reason = "SENSOR_HOLD", "下一步缓冲区的观测不完整"
                    else:
                        state = "MOVE"
                        reason = "沿已观测空闲区接近目标" if route_kind == "GOAL_PATH" else "移动到已观测区域边缘，再从新位置观察"
                        moves.append({"tick": tick, "from": position, "to": nxt,
                                      "buffer_verified": True, "route_kind": route_kind})
                        position, aim = nxt, path[-1]
                        visits[position] += 1
        except (ValueError, TypeError, OverflowError) as exc:
            state, reason = "SENSOR_HOLD", "相机数据无效："+str(exc)
        trace.append({"tick": tick, "state": state, "reason": reason, "position": position,
                      "observed_from": xyz, "path": path, "layer": grid.layer(3, tick),
                      "observation": info, "vision": preview(frames[0]) if frames and previews and info else None})
        if state != "MOVE":
            break
    else:
        trace.append({**trace[-1], "state": "TIMEOUT", "reason": "已达到观察/移动预算，结束本次离散仿真"})
    return {"start": start, "goal": goal, "altitude_m": 3.5, "voxel_size_m": 1,
            "max_ticks": max_ticks, "intrinsics": asdict(k), "scan_directions": 8,
            "horizontal_margin_cells": 1, "free_ttl_ticks": grid.ttl_ticks,
            "width": width, "height": height, "trace": trace, "moves": moves,
            "terminal_state": trace[-1]["state"], "distance_m": len(moves),
            "observations": len({frame["tick"] for frame in trace}),
            "limitations": "理想深度、已知位姿、静态场景，八方向扫描视为同一时刻；射线稀疏格网，固定高度离散移动。不是连续碰撞保证或飞控。"}
