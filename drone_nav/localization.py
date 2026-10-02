"""来源：本项目原创。有界水平定位误差、连续偏移轨迹及独立几何评价。"""

from dataclasses import asdict, dataclass, replace
from math import ceil, cos, isfinite, pi, sin, sqrt

from .pinhole import finite_vector
from .raycast import Box, render


@dataclass(frozen=True)
class LocalizationBudget:
    """每个水平轴的绝对误差上界；来自实验配置，不是在线估计的置信度。"""

    axis_bound_m: float = 0.0
    goal_tolerance_m: float = .5

    def __post_init__(self):
        if (type(self.axis_bound_m) not in (int,float) or not isfinite(self.axis_bound_m)
                or not 0 <= self.axis_bound_m <= 2):
            raise ValueError('horizontal axis error bound must be between 0 and 2 m')
        if (type(self.goal_tolerance_m) not in (int,float) or not isfinite(self.goal_tolerance_m)
                or self.goal_tolerance_m <= 0):
            raise ValueError('positive finite goal tolerance required')

    def planning_margin(self, radius):
        # 每格 1 米；取能覆盖身体和相对误差的整格缓冲，至少保留旧版的一格。
        return max(1,ceil(radius+2*self.axis_bound_m-.5))

    def arrival_confirmable(self):
        return sqrt(2)*self.axis_bound_m <= self.goal_tolerance_m


@dataclass(frozen=True)
class PositionError:
    """仿真专用真实位置减估计位置；导航器不接收此对象或其取值。"""

    amplitude_m: float = 0.0
    kind: str = 'bias'
    period_steps: float = 16.0

    def __post_init__(self):
        if (type(self.amplitude_m) not in (int,float) or not isfinite(self.amplitude_m)
                or not 0 <= self.amplitude_m <= 2 or self.kind not in ('bias','drift')):
            raise ValueError('invalid simulated horizontal position error')
        if type(self.period_steps) not in (int,float) or not isfinite(self.period_steps) or self.period_steps < 4:
            raise ValueError('drift period must be at least four movements')

    def at(self, tick):
        if type(tick) is not int or tick < 0:
            raise ValueError('error model needs a nonnegative scan index')
        if self.kind == 'bias':
            return (self.amplitude_m,self.amplitude_m,0.0)
        angle = 2*pi*tick/self.period_steps
        return (self.amplitude_m*sin(angle+.7),self.amplitude_m*cos(angle+1.2),0.0)


class MislocalizedCamera:
    """真实位置渲染像素，再只向导航器交付估计位置标签；方向和高度不加误差。"""

    def __init__(self, world, error, record=False):
        self.world, self.error, self.record = world,error,record
        self.frames = []

    def capture(self, pose, intrinsics, tick):
        offset = self.error.at(tick)
        true_pose = replace(pose,position=tuple(a+b for a,b in zip(pose.position,offset)))
        frame,_ = render(self.world,true_pose,intrinsics,tick=tick,seed=1201)
        reported = replace(frame,pose=pose)
        if self.record:
            self.frames.append(asdict(reported))
        return reported


def segment_box_clearance(start, end, box):
    """任意线段到轴对齐盒的精确最小距离；逐区间求分段二次函数最小值。"""
    if not finite_vector(start,3) or not finite_vector(end,3):
        raise ValueError('finite endpoints required')
    direction = tuple(b-a for a,b in zip(start,end))
    cuts = {0.0,1.0}
    for a,d,lo,hi in zip(start,direction,box.low,box.high):
        if d:
            cuts.update(t for t in ((lo-a)/d,(hi-a)/d) if 0 < t < 1)
    cuts = sorted(cuts)
    def distance_sq(t):
        return sum(max(lo-(a+t*d),a+t*d-hi,0)**2
                   for a,d,lo,hi in zip(start,direction,box.low,box.high))
    best = min(distance_sq(t) for t in cuts)
    for left,right in zip(cuts,cuts[1:]):
        midpoint = (left+right)/2
        aa = ab = 0.0
        for a,d,lo,hi in zip(start,direction,box.low,box.high):
            mid = a+midpoint*d
            if mid < lo or mid > hi:
                offset = a-(lo if mid < lo else hi)
                aa += d*d
                ab += d*offset
        if aa:
            t = max(left,min(right,-ab/aa))
            best = min(best,distance_sq(t))
    return sqrt(best)


def audit_localization(world, result, error, budget, body_radius=.25):
    """独立读取真值；中途误差按已走距离线性插值，不在扫描之间瞬间跳动。"""
    true_moves = []
    for move in result['moves']:
        tick = move['tick']
        offsets = (error.at(tick),error.at(tick+1))
        # 本实验不注入运动故障，完整一米段才推进误差序列。
        if not move['motion']['completed_edge']:
            raise ValueError('localization audit currently requires complete movement edges')
        endpoints = [tuple(p[i]+(.5 if i<2 else 0)+o[i] for i in range(3))
                     for p,o in zip(((*move['from'],3.5),(*move['to'],3.5)),offsets)]
        distances = [segment_box_clearance(*endpoints,box) for box in world.surfaces if isinstance(box,Box)]
        boundary = min(min(p[0]-body_radius,world.width_m-p[0]-body_radius,
                           p[1]-body_radius,world.height_m-p[1]-body_radius) for p in endpoints)
        length = sqrt(sum((a-b)**2 for a,b in zip(*endpoints)))
        true_moves.append({'tick':tick,'from':endpoints[0],'to':endpoints[1],
                           'distance_m':length,'minimum_center_to_box_m':min(distances) if distances else None,
                           'collision':any(d <= body_radius for d in distances),'out_of_bounds':boundary<0})
    final_tick = len(result['moves'])
    estimate = result['trace'][-1]['position']
    offset = error.at(final_tick)
    actual = (estimate[0]+.5+offset[0],estimate[1]+.5+offset[1],3.5)
    goal = (result['goal'][0]+.5,result['goal'][1]+.5,3.5)
    distance = sqrt(sum((a-b)**2 for a,b in zip(actual,goal)))
    # 扫描时与运动两端的误差均在区间顶点检查；线性插值不超过轴向极值。
    bound_violations = sum(max(abs(v) for v in error.at(t)[:2]) > budget.axis_bound_m+1e-10
                           for t in range(final_tick+1))
    initial = (result['start'][0]+.5+error.at(0)[0],result['start'][1]+.5+error.at(0)[1],3.5)
    point_hits = sum(segment_box_clearance(p,p,b) <= body_radius
                     for p in (initial,actual) for b in world.surfaces if isinstance(b,Box))
    return {'true_moves':true_moves,'true_start':initial,'true_final':actual,
            'true_distance_m':sum(m['distance_m'] for m in true_moves),
            'final_goal_error_m':distance,'inside_goal_tolerance':distance<=budget.goal_tolerance_m,
            'false_arrival':result['terminal_state']=='ARRIVED_WAYPOINT' and distance>budget.goal_tolerance_m,
            'collision_segments':sum(m['collision'] for m in true_moves),'endpoint_collisions':point_hits,
            'boundary_violations':sum(m['out_of_bounds'] for m in true_moves),
            'declared_bound_violations':bound_violations,
            'scope':'给定平移误差与线性插值下的球体/静态长方体评价；不含姿态误差、风或真实控制。'}
