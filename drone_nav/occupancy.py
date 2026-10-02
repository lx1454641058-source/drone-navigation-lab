"""来源：本项目原创。理想深度射线的三维体素观测；通用算法来源见 SOURCES.md。

体素是小立方格。射线穿越仅提供离散空闲证据，不证明格内每一点均无障碍。
当前针对静态场景：占用证据保留，空闲证据过期后变回未知。
"""

from collections import Counter
from math import floor, isfinite

from .pinhole import PerspectiveFrame, Vec3, finite_vector, reconstruct

Voxel = tuple[int, int, int]


def ray_voxels(origin: Vec3, endpoint: Vec3) -> list[Voxel]:
    """1 米格网的逐边界遍历，返回有限线段经过的体素；同时跨越的轴一起推进。"""
    if not finite_vector(origin, 3) or not finite_vector(endpoint, 3):
        raise ValueError("ray coordinates must be finite")
    direction = tuple(b-a for a, b in zip(origin, endpoint))
    cell = [floor(x) for x in origin]
    target = tuple(floor(x) for x in endpoint)
    if sum(abs(a-b) for a, b in zip(cell, target)) > 10000:
        raise ValueError("ray exceeds traversal budget")
    step = [1 if d > 0 else -1 if d < 0 else 0 for d in direction]
    next_t, delta = [], []
    for i, d in enumerate(direction):
        boundary = cell[i]+1 if d > 0 else cell[i]
        next_t.append((boundary-origin[i])/d if d else float("inf"))
        delta.append(abs(1/d) if d else float("inf"))
    traversed = []
    for _ in range(10004):
        traversed.append(tuple(cell))
        if tuple(cell) == target:
            break
        t = min(next_t)
        if t > 1:
            break
        for axis in range(3):
            if abs(next_t[axis]-t) < 1e-10:
                cell[axis] += step[axis]
                next_t[axis] += delta[axis]
    return traversed


class VoxelMap:
    def __init__(self, width: int, height: int, layers: int = 8, ttl_ticks: int = 12):
        if any(type(v) is not int or v < 1 for v in (width, height, layers, ttl_ticks)):
            raise ValueError("positive integer map dimensions and lifetime required")
        self.width, self.height, self.layers, self.ttl_ticks = width, height, layers, ttl_ticks
        self.free_seen: dict[Voxel, int] = {}
        self.occupied: set[Voxel] = set()
        self.last_tick = -1

    def inside(self, cell: Voxel) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height and 0 <= cell[2] < self.layers

    def integrate(self, frames: list[PerspectiveFrame], tick: int, position: Vec3) -> dict:
        """整批先校验再更新；坏帧不能留下半批地图。调用者保证相机标定可信。"""
        if type(tick) is not int or tick <= self.last_tick or not frames or not finite_vector(position, 3):
            raise ValueError("invalid observation batch or non-increasing time")
        if not self.inside(tuple(floor(x) for x in position)):
            raise ValueError("camera outside map")
        for frame in frames:
            frame.validate()
            if frame.tick != tick or any(abs(a-b) > 1e-8 for a, b in zip(frame.pose.position, position)):
                raise ValueError("stale/future frame or inconsistent camera position")
        return self._integrate_rays(frames,tick)

    def integrate_moving(self,frames: list[PerspectiveFrame],tick: int) -> dict:
        """允许一批图像来自不同已知位姿；每根射线使用所属图像的实际原点。"""
        if type(tick) is not int or tick<=self.last_tick or not frames:
            raise ValueError('invalid moving-camera batch')
        for frame in frames:
            frame.validate()
            if frame.tick!=tick or not self.inside(tuple(floor(v) for v in frame.pose.position)):
                raise ValueError('stale/future moving frame or camera outside map')
        return self._integrate_rays(frames,tick)

    def _integrate_rays(self,frames,tick):
        hits: set[Voxel] = set()
        free_votes: Counter[Voxel] = Counter()
        valid = total = 0
        for frame in frames:
            position = frame.pose.position
            total += len(frame.depth_z_m)
            for _, point in reconstruct(frame):
                direction = tuple(b-a for a, b in zip(position, point))
                length_sq = sum(d*d for d in direction)
                if not isfinite(length_sq) or length_sq > 30.0001**2:
                    raise ValueError("depth exceeds configured 30 metre range")
                valid += 1
                # 整数边界上的命中同时保护表面两侧，避免把墙前格擦成空闲。
                endpoint_cells = {tuple(floor(p+sign*d*1e-7) for p, d in zip(point, direction))
                                  for sign in (-1, 0, 1)}
                hits.update(cell for cell in endpoint_cells if self.inside(cell))
                for cell in set(ray_voxels(position, point)) - endpoint_cells:
                    if self.inside(cell):
                        free_votes[cell] += 1
        # 即使其他射线穿过同格，命中证据优先。静态占用不被后续空闲清除。
        self.occupied.update(hits)
        for cell, count in free_votes.items():
            if count >= 2 and cell not in self.occupied:
                self.free_seen[cell] = tick
        self.last_tick = tick
        return {"valid_depth_fraction": valid/max(total, 1), "rays": valid,
                "occupied_voxels": len(self.occupied),
                "free_voxels": sum(self.state(cell, tick) == "free" for cell in self.free_seen)}

    def state(self, cell: Voxel, tick: int) -> str:
        if not self.inside(cell):
            return "occupied"
        if cell in self.occupied:
            return "occupied"
        seen = self.free_seen.get(cell)
        if seen is not None and 0 <= tick-seen <= self.ttl_ticks:
            return "free"
        return "unknown"

    def layer(self, z: int, tick: int) -> list[list[str]]:
        return [[self.state((x, y, z), tick) for x in range(self.width)] for y in range(self.height)]

    def blocked(self, z: int, tick: int) -> set[tuple[int, int]]:
        return {(x, y) for y, row in enumerate(self.layer(z, tick)) for x, value in enumerate(row)
                if value != "free"}
