"""来源：本项目原创。原运动检查的附加采样拒绝条件，不证明整片空间为空。"""
from dataclasses import dataclass
from itertools import product
from math import ceil, floor, sqrt

from .detection_bridge import number, identifier
from .motion import corridor_cells, movement_guard
from .pinhole import PerspectiveFrame


@dataclass(frozen=True)
class SamplingAssumption:
    min_width_m: float
    min_height_m: float
    model: str = 'front_facing_opaque_rectangles'

    def __post_init__(self):
        if any(not number(v) or v<=0 for v in (self.min_width_m,self.min_height_m)):
            raise ValueError('minimum feature sizes must be positive')
        if self.model!='front_facing_opaque_rectangles':
            raise ValueError('unsupported sampling assumption')


@dataclass(frozen=True)
class SamplingView:
    frame: PerspectiveFrame
    frame_id: str
    camera_id: str
    clock_id: str
    captured_at_s: float
    available_at_s: float

    def validate(self):
        if not isinstance(self.frame,PerspectiveFrame):raise ValueError('invalid sampling frame')
        self.frame.validate()
        if not all(identifier(v) for v in (self.frame_id,self.camera_id,self.clock_id)):
            raise ValueError('sampling source identity required')
        if any(not number(v) or v<0 for v in (self.captured_at_s,self.available_at_s)):
            raise ValueError('invalid sampling time')
        if self.available_at_s<self.captured_at_s:raise ValueError('availability precedes capture')


def inspect_voxel(cell, view, assumption, *, depth_error_m=.02, max_range_m=30.):
    """相机覆盖/中心深度/采样密度的必要门槛；不是体积空闲证书。"""
    f=view.frame;k=f.intrinsics
    corners=list(product(*[(v,v+1) for v in cell]))
    projected=[f.pose.project(p,k) for p in corners]
    result=dict(frame_id=view.frame_id,camera_id=view.camera_id,reasons=[])
    if any(p is None for p in projected):
        result['reasons'].append('VOXEL_NOT_FULLY_IN_FRONT');return result
    x0=ceil(min(p[0] for p in projected)-.5-1e-9)
    x1=floor(max(p[0] for p in projected)+.5+1e-9)
    y0=ceil(min(p[1] for p in projected)-.5-1e-9)
    y1=floor(max(p[1] for p in projected)+.5+1e-9)
    far=max(p[2] for p in projected)
    result.update(window=[x0,y0,x1,y1],far_z_m=far,pitch_m=[far/k.fx,far/k.fy])
    if x0<0 or y0<0 or x1>=k.width or y1>=k.height:
        result['reasons'].append('INCOMPLETE_VOXEL_VIEW');return result
    if far/k.fx>assumption.min_width_m+1e-12 or far/k.fy>assumption.min_height_m+1e-12:
        result['reasons'].append('SAMPLING_TOO_COARSE')
    samples=[(u,v,f.depth_z_m[v*k.width+u]) for v in range(y0,y1+1) for u in range(x0,x1+1)]
    result['checked_pixels']=len(samples)
    if any(z is None for _,_,z in samples):result['reasons'].append('MISSING_DEPTH')
    valid=[z for _,_,z in samples if z is not None]
    result['min_depth_z_m']=min(valid) if valid else None
    if any(z-depth_error_m<=far+1e-9 for z in valid):result['reasons'].append('FOREGROUND_OR_DEPTH_MARGIN')
    if any(z is not None and z*sqrt(sum(t*t for t in k.ray(u,v)))>max_range_m for u,v,z in samples):
        result['reasons'].append('OUT_OF_RANGE')
    return result


def sampled_movement_guard(grid,start,end,tick,now_s,stamps,p,config,*,views,assumption=None,
                           clock_id='simulation-seconds',error_bound_m=0.):
    if (not identifier(clock_id) or not number(now_s) or now_s<0 or
        not isinstance(views,(list,tuple)) or len(views)>128):raise ValueError('invalid sampling consumer')
    if assumption is not None and not isinstance(assumption,SamplingAssumption):raise ValueError('invalid assumption')
    ids=set()
    for view in views:
        if not isinstance(view,SamplingView):raise ValueError('invalid sampling view')
        view.validate();key=(view.camera_id,view.frame_id)
        if key in ids:raise ValueError('duplicate sampling frame identity')
        ids.add(key)
    base=movement_guard(grid,start,end,tick,now_s,stamps,p,config,error_bound_m=error_bound_m)
    cells=sorted(corridor_cells(start,(end[0]-start[0],end[1]-start[1]),base['required_distance_m'],
                                config.body_radius_m+2*error_bound_m))
    findings=[]
    for cell in cells:
        checks=[];supported=False
        for view in views:
            reasons=[];frame=view.frame
            if view.clock_id!=clock_id:reasons.append('CLOCK_MISMATCH')
            if view.available_at_s>now_s:reasons.append('FRAME_NOT_AVAILABLE')
            if frame.tick>tick:reasons.append('FUTURE_FRAME_TICK')
            if grid.free_seen.get(cell)!=frame.tick:reasons.append('MAP_EVIDENCE_BATCH_MISMATCH')
            stamp=stamps.get(frame.tick)
            if stamp is None or abs(stamp-view.captured_at_s)>1e-9:reasons.append('MAP_CAPTURE_TIME_MISMATCH')
            if base['latest_stop_s']-view.captured_at_s>config.free_ttl_s+1e-9:
                reasons.append('SAMPLING_EXPIRES_BEFORE_STOP')
            if assumption is None:reasons.append('MINIMUM_FEATURE_ASSUMPTION_UNKNOWN')
            check=dict(frame_id=view.frame_id,camera_id=view.camera_id,reasons=reasons)
            if not reasons:check=inspect_voxel(cell,view,assumption)
            checks.append(check)
            if not check['reasons']:supported=True
        findings.append(dict(voxel=list(cell),supported=supported,views=checks))
    allowed=base['allowed'] and all(v['supported'] for v in findings)
    return dict(allowed=allowed,reason=base['reason'] if not base['allowed'] else
                'SAMPLING_CONDITION_HOLD' if not allowed else 'CONDITIONAL_SIMULATION_APPROVAL',
                baseline=base,required_cells=[list(c) for c in cells],sampling=findings,
                assumption=None if assumption is None else dict(model=assumption.model,min_width_m=assumption.min_width_m,
                    min_height_m=assumption.min_height_m),flight_authorized=False,
                limitation='附加必要门槛；声明的最小尺寸可能错误，点深度不证明整格空闲。')
