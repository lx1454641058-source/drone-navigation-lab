"""来源：本项目原创。理想像素角域深度下界的临时解除视图，仅供研究。

点采样深度不证明像素之间为空。只有显式提供整个像素角域距离下界的
输入才进入几何判断；本轮该契约仅由解析仿真生成器满足，不授权飞行。
"""
from dataclasses import dataclass, replace
from itertools import product
from math import ceil, floor, sqrt

from .detection_bridge import BridgeConfig, number, project_detections
from .observation_channel import shadow_route
from .planning import astar


FOOTPRINT_MIN = 'conservative_pixel_footprint_min_z'


@dataclass(frozen=True)
class ClearanceConfig:
    spatial_margin_m: float = .1
    pixel_margin: float = 1.0
    depth_error_m: float = .05

    def __post_init__(self):
        if any(not number(v) or v < 0 for v in (
                self.spatial_margin_m, self.pixel_margin, self.depth_error_m)):
            raise ValueError('clearance bounds must be finite and nonnegative')


def coverage_window(voxel, context, config):
    """凸盒在正深度下的投影极值位于角点；矩形保守覆盖其整个像。"""
    if len(voxel) != 3 or any(type(v) is not int or v < 0 for v in voxel):
        raise ValueError('invalid voxel')
    k = context.intrinsics
    axes = [(v-config.spatial_margin_m, v+1+config.spatial_margin_m) for v in voxel]
    corners = [context.pose.project(p, k) for p in product(*axes)]
    if any(p is None for p in corners):
        return dict(reason='NOT_FULLY_IN_FRONT')
    # 包含与投影矩形相接的像素角域，不仅是矩形内的像素中心。
    x0 = ceil(min(p[0] for p in corners)-config.pixel_margin-.5-1e-9)
    x1 = floor(max(p[0] for p in corners)+config.pixel_margin+.5+1e-9)
    y0 = ceil(min(p[1] for p in corners)-config.pixel_margin-.5-1e-9)
    y1 = floor(max(p[1] for p in corners)+config.pixel_margin+.5+1e-9)
    if x0 < 0 or y0 < 0 or x1 >= k.width or y1 >= k.height:
        return dict(reason='INCOMPLETE_VIEW')
    return dict(reason=None, window=[x0,y0,x1,y1], far_z_m=max(p[2] for p in corners))


def assess_clearance(snapshot, packet, context, *, now_s, sampling_model='point_samples',
                     bridge_config=None, clearance_config=None):
    """只读快照；解除不能流回原缓存。所有证据每次按原采集时刻重验。"""
    b = bridge_config or BridgeConfig()
    c = clearance_config or ClearanceConfig()
    # 保留对原包的结构检查，但解除仅依赖独立深度几何，不依赖检测阴性。
    packet.validate()
    checked = project_detections(replace(packet, detections=()), context, now_s=now_s,
                                 now_clock_id=snapshot['clock_id'], config=b)
    reasons = list(checked['reasons'])
    if sampling_model != FOOTPRINT_MIN:
        reasons.append('POINT_SAMPLES_CANNOT_CLEAR_VOLUME')
    cells = []
    for old in snapshot['cells']:
        entry = dict(voxel=list(old['voxel']), cleared=False, reasons=list(reasons),
                     old_captured_at_s=old['captured_at_s'])
        if not reasons:
            earliest = min(packet.captured_at_s, context.depth_at_s, context.pose_at_s)
            if earliest <= old['captured_at_s']+b.max_sync_error_s+1e-9:
                entry['reasons'].append('NOT_NEWER_THAN_OLD_EVIDENCE')
            else:
                area = coverage_window(old['voxel'], context, c)
                entry.update(area)
                if area['reason']:
                    entry['reasons'].append(area['reason'])
                else:
                    x0,y0,x1,y1 = area['window']; k = context.intrinsics
                    depths = [(u,v,context.depth_z_m[v*k.width+u])
                              for v in range(y0,y1+1) for u in range(x0,x1+1)]
                    entry['checked_pixels'] = len(depths)
                    valid = [z for _,_,z in depths if z is not None]
                    entry['min_depth_z_m'] = min(valid) if valid else None
                    if len(valid) != len(depths):
                        entry['reasons'].append('MISSING_DEPTH')
                    if any(z is not None and z*sqrt(sum(t*t for t in k.ray(u,v))) > b.max_range_m
                           for u,v,z in depths):
                        entry['reasons'].append('OUT_OF_RANGE')
                    if any(z-c.depth_error_m <= area['far_z_m']+1e-9 for z in valid):
                        entry['reasons'].append('FOREGROUND_OR_INSUFFICIENT_CLEARANCE')
                    entry['cleared'] = not entry['reasons']
        cells.append(entry)
    return dict(cells=cells, context_reasons=reasons, sampling_model=sampling_model,
                captured_at_s=packet.captured_at_s, valid_until_s=checked['valid_until_s'],
                frame_id=packet.frame_id, camera_id=packet.camera_id, sources=checked['sources'],
                flight_authorized=False)


def reobserved_route(grid, channel, packet, context, *, sampling_model='point_samples',
                     clearance_config=None, layer, tick, now_s, start, goal):
    # 几何输入先校验；异常批次不推进原通道时间。
    packet.validate()
    if context is not None:
        context.validate()
    base = shadow_route(grid, channel, layer=layer, tick=tick, now_s=now_s, start=start, goal=goal)
    evidence = assess_clearance(base['observation'], packet, context, now_s=now_s,
                               sampling_model=sampling_model, bridge_config=channel.config,
                               clearance_config=clearance_config)
    retained = {(v['voxel'][0],v['voxel'][1]) for v in evidence['cells']
                if v['voxel'][2] == layer and not v['cleared']}
    blocked = {tuple(v) for v in base['base_blocked']}
    route = astar(grid.width, grid.height, start, goal, blocked | retained, margin=0)
    return dict(before=base, clearance=evidence, restrictions=[list(v) for v in sorted(retained)],
                route=route, status='ROUTE_FOR_REVIEW' if route else 'HOLD', flight_authorized=False)
