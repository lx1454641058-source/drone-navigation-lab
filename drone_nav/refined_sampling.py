"""来源：本项目原创。细分空间、多帧条件覆盖及显式起始空闲声明。"""
from dataclasses import dataclass
from itertools import product
from math import ceil, floor, sqrt, atan2, cos, sin, pi

from .detection_bridge import number, identifier
from .guarded_mission import GuardedMissionVehicle
from .physical_rescan import PhysicalSamplingHistory
from .physical_vehicle import physical_guard, spatial_cells
from .pinhole import Intrinsics, Pose, finite_vector
from .sampling_motion import SamplingAssumption, SamplingView
from dataclasses import replace


@dataclass(frozen=True)
class InitialClearRegion:
    """外部给定的开发假设，不由空检测或世界几何自动生成。"""
    low: tuple
    high: tuple
    captured_at_s: float
    valid_until_s: float
    source: str
    clock_id: str = 'physics-seconds'

    def __post_init__(self):
        if (not finite_vector(self.low,3) or not finite_vector(self.high,3)
                or any(a>=b for a,b in zip(self.low,self.high))
                or not number(self.captured_at_s) or not number(self.valid_until_s)
                or not 0<=self.captured_at_s<self.valid_until_s
                or not identifier(self.source) or not identifier(self.clock_id)):
            raise ValueError('invalid explicit initial clear region')


def inspect_box(low, high, view, assumption):
    """原整格条件推广到轴对齐小盒；深度仍是中心样本，不提供连续空间证明。"""
    frame=view.frame; k=frame.intrinsics
    projected=[frame.pose.project(p,k) for p in product(*zip(low,high))]
    if any(p is None for p in projected):return 'NOT_FULLY_IN_FRONT'
    x0=ceil(min(p[0] for p in projected)-.5-1e-9);x1=floor(max(p[0] for p in projected)+.5+1e-9)
    y0=ceil(min(p[1] for p in projected)-.5-1e-9);y1=floor(max(p[1] for p in projected)+.5+1e-9)
    if x0<0 or y0<0 or x1>=k.width or y1>=k.height:return 'INCOMPLETE_VIEW'
    far=max(p[2] for p in projected)
    if far/k.fx>assumption.min_width_m+1e-12 or far/k.fy>assumption.min_height_m+1e-12:return 'SAMPLING_TOO_COARSE'
    for v in range(y0,y1+1):
        for u in range(x0,x1+1):
            z=frame.depth_z_m[v*k.width+u]
            if z is None:return 'MISSING_DEPTH'
            if z-.02<=far+1e-9:return 'FOREGROUND_OR_DEPTH_MARGIN'
            if z*sqrt(sum(t*t for t in k.ray(u,v)))>30.:return 'OUT_OF_RANGE'
    return None


class RefinedSamplingHistory(PhysicalSamplingHistory):
    def __init__(self, *, divisions=4, initial_region=None):
        if type(divisions) is not int or not 1<=divisions<=4:raise ValueError('divisions must be 1..4')
        if initial_region is not None and not isinstance(initial_region,InitialClearRegion):raise ValueError('invalid initial region')
        super().__init__(clock_id='physics-seconds')
        self.divisions=divisions;self.initial_region=initial_region

    def check(self,start,end,*,now_s,budget,assumption=None):
        self._time(now_s)
        if assumption is not None and not isinstance(assumption,SamplingAssumption):raise ValueError('invalid sampling assumption')
        tick=max(0,self.grid.last_tick)
        base=physical_guard(self.grid,start,end,tick,now_s,self.stamps,budget)
        cells=sorted(spatial_cells(start,end,budget.body_radius_m+budget.tracking_margin_m+budget.stop_reserve_m))
        live=[v for v,_ in self._entries if v.available_at_s<=now_s and
              base['latest_stop_s']<=v.captured_at_s+budget.free_ttl_s+1e-9]
        seed=self.initial_region
        seed_live=(seed is not None and seed.clock_id==self.clock_id and seed.captured_at_s<=now_s and
                   base['latest_stop_s']<=seed.valid_until_s+1e-9)
        rows=[];n=self.divisions
        for cell in cells:
            patches=[]
            for index in product(range(n),repeat=3):
                low=tuple(c+i/n for c,i in zip(cell,index));high=tuple(x+1/n for x in low)
                source=None;reason_counts={}
                # A coarse occupied cell always wins over the external assumption.
                if cell in self.grid.occupied:
                    reason_counts['OBSERVED_OCCUPIED']=1
                elif assumption is None:
                    reason_counts['MINIMUM_FEATURE_ASSUMPTION_UNKNOWN']=1
                elif seed_live and all(a>=b and c<=d for a,b,c,d in zip(low,seed.low,high,seed.high)):
                    source=dict(kind='initial_assumption',source=seed.source,captured_at_s=seed.captured_at_s,
                                valid_until_s=seed.valid_until_s)
                else:
                    for view in live:
                        reason=inspect_box(low,high,view,assumption)
                        if reason is None:
                            source=dict(kind='camera',frame_id=view.frame_id,captured_at_s=view.captured_at_s,
                                        valid_until_s=view.captured_at_s+budget.free_ttl_s)
                            break
                        reason_counts[reason]=reason_counts.get(reason,0)+1
                    if not live:reason_counts['NO_FRESH_FRAME']=1
                patches.append(dict(low=low,high=high,source=source,reasons=reason_counts if source is None else {}))
            rows.append(dict(voxel=list(cell),supported=all(p['source'] is not None for p in patches),patches=patches))
        allowed=base['allowed'] and all(row['supported'] for row in rows)
        self._now=now_s
        return dict(allowed=allowed,baseline=base,required_cells=[list(c) for c in cells],sampling=rows,
                    reason=base['reason'] if not base['allowed'] else 'CONDITIONAL_REFINED_APPROVAL' if allowed else 'REFINED_COVERAGE_HOLD',
                    divisions=n,initial_region_active=bool(seed_live),flight_authorized=False)


class RefinedMissionVehicle(GuardedMissionVehicle):
    def __init__(self,*args,divisions=4,initial_region=None,upper_scan=False,**kwargs):
        # Validate before allocating the native physics instance.
        history=RefinedSamplingHistory(divisions=divisions,initial_region=initial_region)
        if type(upper_scan) is not bool:raise ValueError('upper_scan must be bool')
        super().__init__(*args,**kwargs)
        self.sampling_history=history;self.upper_scan=upper_scan

    def scan(self,tick,aim):
        frames=super().scan(tick,aim)
        if not self.upper_scan:return frames
        k=Intrinsics(40,30,25,25,19.5,14.5);extra=[]
        for i in range(8):
            state=self.state()
            def upward(xyz,i=i):
                heading=atan2(aim[1]+.5-xyz[1],aim[0]+.5-xyz[0])+i*pi/4
                return Pose.look_at(xyz,(xyz[0]+8*cos(heading),xyz[1]+8*sin(heading),xyz[2]+3))
            frame=self.capture(k,tick,upward);extra.append((state,frame));frames.append(frame)
            self.hold(.1)
        self.hold(.1);available=self.state()['time_s']
        for state,frame in extra:
            seq=self.sampling_history.grid.last_tick+1
            self.sampling_history.add(SamplingView(replace(frame,tick=seq),f'upper-{seq}','camera',
                'physics-seconds',state['time_s'],available),now_s=available)
            self.source_frames.append(dict(navigation_tick=tick,ledger_tick=seq,
                captured_at_s=state['time_s'],available_at_s=available))
        return frames
