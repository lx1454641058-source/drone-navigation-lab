"""来源：本项目原创程序化图像生成器，不复制照片/模型/纹理。

这是具有已知比例的正交俯视 RGB-D 相机模型，不是 Gazebo 或透视光学仿真。
此模块可以读取场景真值；分类器及图像分析模块不能读取本模块的真值标签。
"""

from math import tan, radians
from random import Random

from .imaging import RGB, RGBDFrame
from .mission import Scenario

PALETTE = ((155, 151, 143), (65, 133, 58), (40, 111, 178), (148, 67, 49), (218, 173, 32))


def painted_pixel(label: int, rng: Random, light: float = 1.0) -> RGB:
    base = PALETTE[label] if label >= 0 else (208, 35, 197)
    shade = light*rng.uniform(0.96, 1.04)
    return tuple(max(0, min(255, int(c*shade+rng.gauss(0, 3)))) for c in base)


def training_image(seed: int, size: int = 40, light: float | None = None) -> tuple[list[RGB], list[int]]:
    """每张样本的布局、色彩扰动由独立种子确定；标签只给训练和评估。"""
    rng = Random(seed)
    light = rng.uniform(0.78, 1.16) if light is None else light
    offset = rng.randrange(5)
    labels = [((x//8) + (y//8) + offset) % 5 for y in range(size) for x in range(size)]
    pixels = [painted_pixel(label, rng, light) for label in labels]
    return pixels, labels


class OrthographicCamera:
    def __init__(self, scenario: Scenario, seed: int = 700, pixels_per_cell: int = 3):
        if pixels_per_cell < 2:
            raise ValueError("camera needs at least two pixels per cell")
        self.scenario = scenario
        self.seed = seed
        self.ppc = pixels_per_cell

    def capture(self, position: tuple[int, int], tick: int) -> RGBDFrame:
        s = self.scenario
        radius = s.sensor_radius_cells
        cells = radius*2+1
        size = cells*self.ppc
        origin = (position[0]-radius, position[1]-radius)
        rgb: list[RGB] = []
        depth: list[float] = []
        for y in range(size):
            for x in range(size):
                gx, gy = origin[0]+x//self.ppc, origin[1]+y//self.ppc
                wx, wy = origin[0]+(x+0.5)/self.ppc, origin[1]+(y+0.5)/self.ppc
                if not (0 <= gx < s.width and 0 <= gy < s.height):
                    rgb.append((0, 0, 0)); depth.append(float("nan")); continue
                # 固定世界纹理，视角移动时同一像素不随机换色；不同场景种子独立。
                rng = Random(self.seed + (gy*self.ppc+y%self.ppc)*10007 + gx*self.ppc+x%self.ppc)
                label, height = 0, s.terrain.get((gx, gy), 0.0)
                for site in s.sites:
                    dx, dy = wx-(site.cell[0]+0.5), wy-(site.cell[1]+0.5)
                    if max(abs(dx), abs(dy)) <= 3.5:
                        label = {"paved": 0, "landing_pad": 0, "water": 2,
                                 "grass": 1}.get(site.surface, -1)
                        height = 0.9 + tan(radians(site.slope_deg))*dx
                        if max(abs(dx), abs(dy)) > site.clear_radius_m:
                            label, height = 3, 2.0
                        if site.occupied and abs(dx) < 0.8 and abs(dy) < 0.8:
                            label, height = 4, 2.0
                if (gx, gy) in s.known_obstacles or (gx, gy) in s.hidden_obstacles:
                    label, height = 3, 7.0
                if s.terrain.get((gx, gy), 0) > 0:
                    label, height = 1, s.terrain[(gx, gy)]
                rgb.append(painted_pixel(label, rng))
                depth.append(s.cruise_altitude_m-height+rng.gauss(0, 0.006))
        if s.degraded_tick == tick:
            rgb = [(0, 0, 0)]*len(rgb)
            depth = [float("nan")]*len(depth)
        return RGBDFrame(size, size, tuple(rgb), tuple(depth), origin, self.ppc,
                         s.cruise_altitude_m, tick)
