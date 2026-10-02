"""来源：本项目原创。解析射线与平面/长方体相交，选择最近表面产生遮挡。

只用于理想针孔 RGB-D 传感器开发；不模拟飞行、反射、阴影和透明水体。
场景真值仅在此渲染模块及独立评价器使用。
"""

from dataclasses import dataclass
from math import floor, sin, sqrt
from random import Random

from .pinhole import Intrinsics, PerspectiveFrame, Pose, Vec3, dot, finite_vector
from .rendering import painted_pixel


@dataclass(frozen=True)
class Surface:
    name: str
    label: int
    bounds: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    z0: float = 0.0
    slope_x: float = 0.0
    slope_y: float = 0.0

    def __post_init__(self):
        if (not finite_vector(self.bounds, 4) or self.bounds[0] >= self.bounds[1]
                or self.bounds[2] >= self.bounds[3]
                or not finite_vector((self.z0, self.slope_x, self.slope_y), 3)):
            raise ValueError("invalid surface geometry")

    def intersect(self, origin: Vec3, direction: Vec3) -> float | None:
        # z = z0 + slope_x*x + slope_y*y；射线参数直接等于相机 z 深度。
        denominator = direction[2]-self.slope_x*direction[0]-self.slope_y*direction[1]
        if abs(denominator) < 1e-12:
            return None
        t = (self.z0+self.slope_x*origin[0]+self.slope_y*origin[1]-origin[2])/denominator
        if t <= 1e-7:
            return None
        x, y = origin[0]+t*direction[0], origin[1]+t*direction[1]
        return t if self.bounds[0] <= x <= self.bounds[1] and self.bounds[2] <= y <= self.bounds[3] else None


@dataclass(frozen=True)
class Box:
    name: str
    label: int
    low: Vec3
    high: Vec3

    def __post_init__(self):
        if (not finite_vector(self.low, 3) or not finite_vector(self.high, 3)
                or any(a >= b for a, b in zip(self.low, self.high))):
            raise ValueError("invalid box geometry")

    def intersect(self, origin: Vec3, direction: Vec3) -> float | None:
        near, far = -float("inf"), float("inf")
        for o, d, low, high in zip(origin, direction, self.low, self.high):
            if abs(d) < 1e-12:
                if not low <= o <= high:
                    return None
                continue
            a, b = (low-o)/d, (high-o)/d
            near, far = max(near, min(a, b)), min(far, max(a, b))
            if near > far:
                return None
        distance = near if near > 1e-7 else far
        return distance if distance > 1e-7 else None


@dataclass(frozen=True)
class World:
    surfaces: tuple[Surface | Box, ...]
    width_m: int = 20
    height_m: int = 16


def demo_world() -> World:
    return World((Surface("ground", 0, (0, 20, 0, 16)),
                  Surface("water", 2, (12, 18, 1, 5), .015),
                  Surface("unknown", -1, (14, 18, 11, 15), .02),
                  Surface("ramp", 0, (2, 6, 10, 14), -.3, .15),
                  Box("building", 3, (8, 6, 0), (10, 10, 5)),
                  Box("vegetation", 1, (3, 2, 0), (5, 4, 2.5)),
                  Box("person", 4, (14, 7.5, 0), (14.6, 8.1, 1.7))))


def first_hit(world: World, origin: Vec3, direction: Vec3):
    nearest, found = float("inf"), None
    for surface in world.surfaces:
        t = surface.intersect(origin, direction)
        if t is not None and t < nearest:
            nearest, found = t, surface
    return (nearest, found) if found is not None else None


@dataclass(frozen=True)
class RenderTruth:
    """仅供评价，永远不传给 reconstruct 或颜色分类器。"""
    points: tuple[Vec3 | None, ...]
    labels: tuple[int, ...]
    objects: tuple[str | None, ...]


def render(world: World, pose: Pose, k: Intrinsics, *, seed: int = 901,
           light: float = 1.0, dropout: float = 0.0, noise_std_m: float = 0.0,
           max_range_m: float = 30.0, tick: int = 0) -> tuple[PerspectiveFrame, RenderTruth]:
    if (not finite_vector((light, dropout, noise_std_m, max_range_m), 4)
            or light < 0 or not 0 <= dropout <= 1 or noise_std_m < 0 or max_range_m <= 0
            or type(seed) is not int or type(tick) is not int or tick < 0):
        raise ValueError("invalid rendering parameters")
    rgb, depths, points, labels, objects = [], [], [], [], []
    rng = Random(seed)
    for v in range(k.height):
        for u in range(k.width):
            direction = pose.rotate(k.ray(u, v))
            hit = first_hit(world, pose.position, direction)
            if hit is None or hit[0]*sqrt(dot(direction, direction)) > max_range_m:
                rgb.append((18, 27, 40)); depths.append(None)
                points.append(None); labels.append(-1); objects.append(None)
                continue
            z, surface = hit
            point = tuple(o+z*d for o, d in zip(pose.position, direction))
            # 世界坐标固定的纹理，与图像像素位置无关；颜色仍是简化人工材质。
            texture = .92 + .045*sin(point[0]*7) + .035*sin(point[1]*11)
            noise = Random(seed+floor(point[0]*20)*100003+floor(point[1]*20)*101+floor(point[2]*20))
            rgb.append(painted_pixel(surface.label, noise, light*texture))
            measured = z+rng.gauss(0, noise_std_m)
            depths.append(None if rng.random() < dropout or measured <= 0 else measured)
            points.append(point); labels.append(surface.label); objects.append(surface.name)
    frame = PerspectiveFrame(k, pose, tuple(rgb), tuple(depths), tick)
    frame.validate()
    return frame, RenderTruth(tuple(points), tuple(labels), tuple(objects))
