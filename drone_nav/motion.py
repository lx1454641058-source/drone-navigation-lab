"""来源：本项目原创。一维加速/匀速/减速及延迟制动，属于运动学而非飞行动力学。"""

from dataclasses import dataclass
from math import floor, isfinite, sqrt


@dataclass(frozen=True)
class MotionConfig:
    max_speed_mps: float = 2.0
    acceleration_mps2: float = 1.0
    brake_mps2: float = 1.0
    reaction_s: float = .2
    scan_s: float = .8
    processing_s: float = .1
    free_ttl_s: float = 12.0
    sample_dt_s: float = .05
    body_radius_m: float = .25
    distance_margin_m: float = .15

    def __post_init__(self):
        positive = (self.max_speed_mps, self.acceleration_mps2, self.brake_mps2,
                    self.free_ttl_s, self.sample_dt_s, self.body_radius_m)
        nonnegative = (self.reaction_s, self.scan_s, self.processing_s, self.distance_margin_m)
        if any(type(v) not in (int, float) or not isfinite(v) or v <= 0 for v in positive):
            raise ValueError('motion parameters must be finite and positive')
        if any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in nonnegative):
            raise ValueError('motion delays and margin must be finite and nonnegative')
        if self.body_radius_m >= .5 or self.sample_dt_s > .2:
            raise ValueError('this fixed-layer experiment requires radius < .5 m and dt <= .2 s')


@dataclass(frozen=True)
class Profile:
    distance_m: float
    peak_mps: float
    accelerate_s: float
    cruise_s: float
    brake_s: float
    acceleration: float
    braking: float

    @property
    def total_s(self):
        return self.accelerate_s+self.cruise_s+self.brake_s

    def at(self, time_s):
        t = max(0.0, min(self.total_s, time_s))
        if t <= self.accelerate_s:
            return .5*self.acceleration*t*t, self.acceleration*t
        x0 = .5*self.peak_mps*self.accelerate_s
        t -= self.accelerate_s
        if t <= self.cruise_s:
            return x0+self.peak_mps*t, self.peak_mps
        t -= self.cruise_s
        return (x0+self.peak_mps*self.cruise_s+self.peak_mps*t-.5*self.braking*t*t,
                max(0.0, self.peak_mps-self.braking*t))


def profile(distance_m: float, config: MotionConfig) -> Profile:
    if type(distance_m) not in (int, float) or not isfinite(distance_m) or distance_m <= 0:
        raise ValueError('movement distance must be positive')
    peak = min(config.max_speed_mps,
               sqrt(2*distance_m/(1/config.acceleration_mps2+1/config.brake_mps2)))
    ta, tb = peak/config.acceleration_mps2, peak/config.brake_mps2
    cruise = max(0.0, (distance_m-.5*peak*(ta+tb))/peak)
    return Profile(distance_m, peak, ta, cruise, tb, config.acceleration_mps2, config.brake_mps2)


def simulate(profile: Profile, config: MotionConfig, failure_at_s=None) -> dict:
    """每段从静止出发并结束于静止；感知故障后先保持当前速度，再按约定减速度制动。"""
    failed = failure_at_s is not None
    if failed and (type(failure_at_s) not in (int, float) or not isfinite(failure_at_s)
                   or not 0 <= failure_at_s < profile.total_s):
        raise ValueError('failure time must lie within movement')
    if failed:
        x_fail, v_fail = profile.at(failure_at_s)
        brake_start = failure_at_s+config.reaction_s
        end = brake_start+v_fail/config.brake_mps2
    else:
        x_fail = v_fail = brake_start = None
        end = profile.total_s
    if end/config.sample_dt_s > 100000:
        raise ValueError('movement exceeds sample budget')
    # 显式纳入阶段边界，避免采样间隔掩盖故障或停止时刻。
    times = {0.0, end, min(profile.accelerate_s, end), min(profile.accelerate_s+profile.cruise_s, end)}
    times.update(i*config.sample_dt_s for i in range(floor(end/config.sample_dt_s)+1))
    if failed:
        times.update((failure_at_s, brake_start))
    samples = []
    for t in sorted(times):
        if not failed or t < failure_at_s:
            x, v = profile.at(t)
            state = 'ACCELERATE' if t < profile.accelerate_s else 'CRUISE' if t < profile.accelerate_s+profile.cruise_s else 'BRAKE'
        elif t < brake_start:
            x, v, state = x_fail+v_fail*(t-failure_at_s), v_fail, 'REACTION_DELAY'
        else:
            delta = min(t-brake_start, v_fail/config.brake_mps2)
            x = x_fail+v_fail*config.reaction_s+v_fail*delta-.5*config.brake_mps2*delta*delta
            v, state = max(0.0, v_fail-config.brake_mps2*delta), 'EMERGENCY_BRAKE'
        samples.append({'t_s':t, 'distance_m':x, 'speed_mps':v, 'phase':state})
    samples[-1]['phase'] = 'STOPPED'
    return {'samples':samples, 'duration_s':end, 'distance_m':samples[-1]['distance_m'],
            'failure_at_s':failure_at_s, 'speed_at_failure_mps':v_fail,
            'reaction_distance_m':None if not failed else v_fail*config.reaction_s,
            'braking_distance_m':None if not failed else v_fail*v_fail/(2*config.brake_mps2),
            'completed_edge':not failed}


def corridor_cells(start, direction, distance_m, radius):
    """水平轴向扫掠体的保守包围盒，返回会接触的地图格；不读取真实障碍。"""
    if direction not in ((1,0),(-1,0),(0,1),(0,-1)):
        raise ValueError('only axis-aligned movements supported')
    center = (start[0]+.5, start[1]+.5)
    end = (center[0]+direction[0]*distance_m, center[1]+direction[1]*distance_m)
    return {(x,y,3) for x in range(floor(min(center[0],end[0])-radius),floor(max(center[0],end[0])+radius)+1)
            for y in range(floor(min(center[1],end[1])-radius),floor(max(center[1],end[1])+radius)+1)}


def movement_guard(grid, start, end, tick, now_s, stamp_seconds, p, config, *, error_bound_m=0.0):
    if type(error_bound_m) not in (int,float) or not isfinite(error_bound_m) or not 0 <= error_bound_m <= 2:
        raise ValueError('invalid horizontal position error bound')
    direction = (end[0]-start[0],end[1]-start[1])
    # 对同一减速度的计划刹车，最迟停止位置 <= 计划距离 + 峰值速度*反应延迟。
    required = p.distance_m+p.peak_mps*config.reaction_s+config.distance_margin_m
    latest_stop_s = now_s+p.total_s+config.reaction_s
    # 建图位置误差与当前机体位置误差可能反向，逐轴相对差最多为 2E。
    cells = corridor_cells(start,direction,required,config.body_radius_m+2*error_bound_m)
    unsafe = [cell for cell in cells if grid.state(cell,tick) != 'free']
    stale = [cell for cell in cells if cell in grid.free_seen and
             latest_stop_s-stamp_seconds[grid.free_seen[cell]] > config.free_ttl_s]
    return {'allowed':not unsafe and not stale, 'required_distance_m':required,
            'latest_stop_s':latest_stop_s, 'unsafe_cells':len(unsafe), 'expiring_cells':len(stale),
            'reason':'BRAKE_MARGIN_HOLD' if unsafe else 'STALE_MAP_HOLD' if stale else 'APPROVED'}
