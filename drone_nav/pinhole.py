"""来源：本项目原创。针孔相机、坐标转换及 RGB-D 点云；公式来源见 SOURCES.md。

世界坐标：x 向东、y 向北、z 向上，单位米。
相机坐标：x 向图像右、y 向图像下、z 向镜头前方。
depth_z_m 是相机 z 轴深度，不是沿射线的欧氏距离。
"""

from dataclasses import dataclass
from math import floor, isfinite, sqrt

from .imaging import RGB

Vec3 = tuple[float, float, float]


def dot(a: Vec3, b: Vec3) -> float:
    return sum(x*y for x, y in zip(a, b))


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def finite_vector(values, size: int) -> bool:
    return (isinstance(values, (tuple, list)) and len(values) == size
            and all(type(v) in (float, int) and isfinite(v) for v in values))


def normalize(a: Vec3) -> Vec3:
    length = sqrt(dot(a, a))
    if length < 1e-10:
        raise ValueError("direction must be nonzero")
    return tuple(x/length for x in a)


@dataclass(frozen=True)
class Intrinsics:
    """相机内参：焦距 fx/fy 与图像中心 cx/cy，均以像素为单位。"""

    width: int = 96
    height: int = 72
    fx: float = 70.0
    fy: float = 70.0
    cx: float = 47.5
    cy: float = 35.5

    def __post_init__(self):
        if (type(self.width) is not int or type(self.height) is not int
                or min(self.width, self.height) < 1 or self.width*self.height > 1_000_000
                or not finite_vector((self.fx, self.fy, self.cx, self.cy), 4)
                or min(self.fx, self.fy) <= 0
                or not (-0.5 <= self.cx < self.width and -0.5 <= self.cy < self.height)):
            raise ValueError("invalid camera intrinsics")

    def ray(self, u: float, v: float) -> Vec3:
        return ((u-self.cx)/self.fx, (v-self.cy)/self.fy, 1.0)


@dataclass(frozen=True)
class Pose:
    position: Vec3
    right: Vec3
    down: Vec3
    forward: Vec3

    def __post_init__(self):
        axes = (self.right, self.down, self.forward)
        if not all(finite_vector(v, 3) for v in (self.position, *axes)):
            raise ValueError("pose must be finite")
        if (any(abs(dot(axis, axis)-1) > 1e-6 for axis in axes)
                or any(abs(dot(axes[i], axes[j])) > 1e-6 for i in range(3) for j in range(i))
                or dot(cross(self.right, self.down), self.forward) < 1-1e-6):
            raise ValueError("camera axes must form a right-handed orthonormal basis")

    @classmethod
    def look_at(cls, position: Vec3, target: Vec3) -> "Pose":
        if not finite_vector(position, 3) or not finite_vector(target, 3):
            raise ValueError("position and target must be finite")
        forward = normalize(tuple(b-a for a, b in zip(position, target)))
        up = (0.0, 0.0, 1.0) if abs(forward[2]) < .999 else (0.0, 1.0, 0.0)
        right = normalize(cross(forward, up))
        return cls(position, right, cross(forward, right), forward)

    def rotate(self, camera_vector: Vec3) -> Vec3:
        return tuple(sum(camera_vector[j]*axis[i] for j, axis in enumerate(
            (self.right, self.down, self.forward))) for i in range(3))

    def project(self, point: Vec3, intrinsics: Intrinsics) -> tuple[float, float, float] | None:
        if not finite_vector(point, 3):
            raise ValueError("point must be finite")
        relative = tuple(x-y for x, y in zip(point, self.position))
        z = dot(relative, self.forward)
        if z <= 0:
            return None
        return (intrinsics.fx*dot(relative, self.right)/z+intrinsics.cx,
                intrinsics.fy*dot(relative, self.down)/z+intrinsics.cy, z)

    def unproject(self, u: float, v: float, depth_z_m: float, intrinsics: Intrinsics) -> Vec3:
        if not finite_vector((u, v, depth_z_m), 3) or depth_z_m <= 0:
            raise ValueError("pixel coordinates and positive z-depth must be finite")
        ray = self.rotate(intrinsics.ray(u, v))
        return tuple(origin+direction*depth_z_m for origin, direction in zip(self.position, ray))


@dataclass(frozen=True)
class PerspectiveFrame:
    intrinsics: Intrinsics
    pose: Pose
    rgb: tuple[RGB, ...]
    depth_z_m: tuple[float | None, ...]
    tick: int

    def validate(self):
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("invalid frame tick")
        size = self.intrinsics.width*self.intrinsics.height
        if len(self.rgb) != size or len(self.depth_z_m) != size:
            raise ValueError("frame shape mismatch")
        if any(not isinstance(p, (tuple, list)) or len(p) != 3 or
               any(type(v) is not int or not 0 <= v <= 255 for v in p) for p in self.rgb):
            raise ValueError("invalid RGB pixel")
        if any(d is not None and (type(d) not in (int, float) or not isfinite(d) or d <= 0)
               for d in self.depth_z_m):
            raise ValueError("invalid z-depth; missing values must be None")


def reconstruct(frame: PerspectiveFrame) -> list[tuple[int, Vec3]]:
    """只读取像素、深度与标定；无场景真值入口。保留像素索引以关联分类结果。"""
    frame.validate()
    k = frame.intrinsics
    return [(i, frame.pose.unproject(i % k.width, i // k.width, z, k))
            for i, z in enumerate(frame.depth_z_m) if z is not None]


def observed_grid(points: list[tuple[int, Vec3]], labels: list[int], *, width: int,
                  height: int, resolution_m: float = .5) -> dict:
    """仅记录射线端点的观测，不把穿越射线、遮挡后面或格内稀疏点当成安全空地。"""
    if (type(width) is not int or type(height) is not int or min(width, height) < 1
            or type(resolution_m) not in (int, float) or not isfinite(resolution_m) or resolution_m <= 0):
        raise ValueError("invalid grid geometry")
    cells = {}
    for index, (x, y, z) in points:
        cell = (floor(x/resolution_m), floor(y/resolution_m))
        if not (0 <= cell[0] < width and 0 <= cell[1] < height):
            continue
        item = cells.setdefault(cell, {"x": cell[0], "y": cell[1], "count": 0,
                                      "min_z_m": z, "max_z_m": z, "labels": set()})
        item["count"] += 1
        item["min_z_m"] = min(item["min_z_m"], z)
        item["max_z_m"] = max(item["max_z_m"], z)
        item["labels"].add(labels[index])
    packed = []
    for cell in sorted(cells):
        item = cells[cell]
        item["labels"] = sorted(item["labels"])
        # 0.25 米仅为显示层的“抬高表面”阈值，不是飞行安全判定。
        item["kind"] = ("raised" if item["max_z_m"] > .25 else
                        "uncertain" if -1 in item["labels"] else "surface")
        packed.append(item)
    return {"width": width, "height": height, "resolution_m": resolution_m,
            "unknown_cells": width*height-len(packed), "cells": packed}
