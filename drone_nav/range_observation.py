"""来源：本项目原创。理想测距状态与有界未命中；普通 None 仍表示未知。

仅新增内部开发类型；保留旧帧、重建、地图、控制器和实验行为。
NO_HIT 是模拟传感器明确给出的有限量程结果，不是从真实相机空值推断。
"""
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from math import ceil, floor, sqrt

from .detection_bridge import number
from .pinhole import PerspectiveFrame, finite_vector
from .raycast import render
from .refined_sampling import RefinedSamplingHistory
from .sampling_motion import SamplingAssumption


@dataclass(frozen=True)
class RangeFrame(PerspectiveFrame):
    outcomes: tuple[str, ...]
    max_range_m: float
    sensor_model: str = 'ideal_center_ray'

    def validate(self):
        super().validate()
        if (self.sensor_model != 'ideal_center_ray' or not number(self.max_range_m)
                or not 0 < self.max_range_m <= 30. or type(self.outcomes) is not tuple
                or len(self.outcomes) != len(self.depth_z_m)):
            raise ValueError('invalid range sensor contract')
        k = self.intrinsics
        for i, (outcome, depth) in enumerate(zip(self.outcomes, self.depth_z_m)):
            if outcome not in ('HIT', 'NO_HIT', 'INVALID'):
                raise ValueError('unknown range outcome')
            if (outcome == 'HIT') != (depth is not None):
                raise ValueError('range outcome contradicts depth')
            if depth is not None:
                norm = sqrt(sum(x*x for x in k.ray(i % k.width, i // k.width)))
                if depth*norm > self.max_range_m + 1e-9:
                    raise ValueError('hit outside radial range')


def render_range(world, pose, intrinsics, *, tick=0, seed=1701, max_range_m=30., invalid=False):
    """传感器边界内查询世界；消费者只收到状态、真实命中深度和有限量程。

旧渲染器在无噪声、无丢帧时的 None 才能对应 NO_HIT。全通道故障也作用于
本来没有命中的射线，防止天空在传感器掉线时仍提供空闲证据。
"""
    if type(invalid) is not bool:
        raise ValueError('sensor fault must be explicit bool')
    f, _ = render(world, pose, intrinsics, seed=seed, tick=tick, max_range_m=max_range_m)
    outcomes = tuple('INVALID' if invalid else 'NO_HIT' if d is None else 'HIT'
                     for d in f.depth_z_m)
    result = RangeFrame(f.intrinsics, f.pose, f.rgb,
                        (None,)*len(f.depth_z_m) if invalid else f.depth_z_m,
                        f.tick, outcomes, max_range_m)
    result.validate()
    return result


def legacy_frame(frame):
    """丢弃额外状态时所有未命中仍为 None，不创造虚拟表面或障碍端点。"""
    return PerspectiveFrame(frame.intrinsics, frame.pose, frame.rgb, frame.depth_z_m, frame.tick)


class RangeBoxInspector:
    """对一个已验证来源反复检查小盒；没有世界几何、地图或控制器入口。"""
    def __init__(self, view, assumption):
        view.validate()
        if not isinstance(view.frame, RangeFrame):
            raise ValueError('explicit range outcomes required')
        if assumption is not None and not isinstance(assumption, SamplingAssumption):
            raise ValueError('invalid sampling assumption')
        self.view = deepcopy(view)
        self.assumption = assumption

    def inspect(self, low, high):
        if (not finite_vector(low, 3) or not finite_vector(high, 3)
                or any(a >= b for a, b in zip(low, high))):
            raise ValueError('invalid box')
        f = self.view.frame; k = f.intrinsics; a = self.assumption
        if a is None:
            return 'MINIMUM_FEATURE_ASSUMPTION_UNKNOWN'
        projected = [f.pose.project(p, k) for p in product(*zip(low, high))]
        if any(p is None for p in projected):
            return 'NOT_FULLY_IN_FRONT'
        x0 = ceil(min(p[0] for p in projected)-.5-1e-9)
        x1 = floor(max(p[0] for p in projected)+.5+1e-9)
        y0 = ceil(min(p[1] for p in projected)-.5-1e-9)
        y1 = floor(max(p[1] for p in projected)+.5+1e-9)
        if x0 < 0 or y0 < 0 or x1 >= k.width or y1 >= k.height:
            return 'INCOMPLETE_VIEW'
        far = max(p[2] for p in projected)
        if far/k.fx > a.min_width_m+1e-12 or far/k.fy > a.min_height_m+1e-12:
            return 'SAMPLING_TOO_COARSE'
        for v in range(y0, y1+1):
            for u in range(x0, x1+1):
                i = v*k.width+u
                if f.outcomes[i] == 'INVALID':
                    return 'SENSOR_INVALID'
                # max_range 是沿射线的长度；不能直接当成相机 z 深度。
                norm = sqrt(sum(x*x for x in k.ray(u, v)))
                bound = f.max_range_m/norm if f.outcomes[i] == 'NO_HIT' else f.depth_z_m[i]
                if bound-.02 <= far+1e-9:
                    return 'RANGE_BOUND_OR_FOREGROUND'
        return None


class RangeSamplingHistory(RefinedSamplingHistory):
    """在原完整覆盖与原物理门槛之上，消费显式未命中的有限射线界限。"""
    def check(self, start, end, *, now_s, budget, assumption=None):
        result = super().check(start, end, now_s=now_s, budget=budget, assumption=assumption)
        stop = result['baseline']['latest_stop_s']
        candidates = [RangeBoxInspector(view, assumption) for view, _ in self._entries
                      if isinstance(view.frame, RangeFrame) and view.available_at_s <= now_s
                      and stop <= view.captured_at_s+budget.free_ttl_s+1e-9]
        for row in result['sampling']:
            if tuple(row['voxel']) in self.grid.occupied:
                continue
            for patch in row['patches']:
                if patch['source'] is not None:
                    continue
                for inspector in candidates:
                    if inspector.inspect(patch['low'], patch['high']) is None:
                        view = inspector.view
                        patch['source'] = dict(kind='camera', frame_id=view.frame_id,
                            captured_at_s=view.captured_at_s,
                            valid_until_s=view.captured_at_s+budget.free_ttl_s,
                            range_contract=view.frame.sensor_model,
                            max_range_m=view.frame.max_range_m)
                        patch['reasons'] = {}
                        break
            row['supported'] = all(p['source'] is not None for p in row['patches'])
        result['allowed'] = result['baseline']['allowed'] and all(r['supported'] for r in result['sampling'])
        result['reason'] = (result['baseline']['reason'] if not result['baseline']['allowed'] else
                            'CONDITIONAL_RANGE_APPROVAL' if result['allowed'] else 'RANGE_COVERAGE_HOLD')
        return result
