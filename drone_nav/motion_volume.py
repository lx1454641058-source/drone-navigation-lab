"""来源：本项目原创。候选直线运动的停止范围、深度检查与局部规划对照。"""
from dataclasses import asdict,dataclass
from math import sqrt
import re

from .depth_volume import DepthVolumeInspector,QueryVolume
from .detection_bridge import identifier,number
from .motion import MotionConfig,profile
from .planning import astar


@dataclass(frozen=True)
class RestMotionCandidate:
    name: str
    world_frame: str
    reference_fingerprint: str
    start: tuple
    end: tuple

    def __post_init__(self):
        if not all(identifier(v) for v in (self.name,self.world_frame)) or type(self.reference_fingerprint) is not str or not re.fullmatch('[0-9a-f]{64}',self.reference_fingerprint):
            raise ValueError('candidate identity and full reference fingerprint required')
        if any(type(p) is not tuple or len(p)!=3 or not all(number(v) and abs(v)<=1000 for v in p) for p in (self.start,self.end)):
            raise ValueError('immutable finite candidate coordinates required')
        if not 1e-6<=sqrt(sum((b-a)**2 for a,b in zip(self.start,self.end)))<=20:
            raise ValueError('candidate length outside research limit')


def stopping_volume(candidate,config,*,now_s,error_bound_m):
    """3-D AABB encloses the model's full body sweep, including emergency stops.

    Assumes rest at start, straight-line motion and constant speed during the
    reaction delay. No wind, lateral drift, acceleration during delay or hardware
    braking guarantee is inferred. Per-axis map/body errors may oppose: 2E.
    """
    if type(candidate) is not RestMotionCandidate or type(config) is not MotionConfig:
        raise ValueError('validated motion candidate/configuration required')
    if not number(now_s) or now_s<0 or not number(error_bound_m) or not 0<=error_bound_m<=2:
        raise ValueError('finite time and declared error bound required')
    delta=tuple(b-a for a,b in zip(candidate.start,candidate.end))
    distance=sqrt(sum(v*v for v in delta)); direction=tuple(v/distance for v in delta)
    p=profile(distance,config)
    # Reuse the established model's D + peak*tau bound, not v_max^2/(2b)
    # added to D (which would count the planned braking distance twice).
    required=distance+p.peak_mps*config.reaction_s+config.distance_margin_m
    endpoint=tuple(a+d*required for a,d in zip(candidate.start,direction))
    padding=config.body_radius_m+2*error_bound_m
    volume=QueryVolume(candidate.name,candidate.world_frame,
        tuple(min(a,b)-padding for a,b in zip(candidate.start,endpoint)),
        tuple(max(a,b)+padding for a,b in zip(candidate.start,endpoint)))
    envelope=dict(candidate=asdict(candidate),profile=asdict(p),normal_duration_s=p.total_s,
        direction=list(direction),required_distance_m=required,extended_endpoint=list(endpoint),
        relative_padding_m=padding,error_bound_m=error_bound_m,config=asdict(config),volume=asdict(volume),
        departure_at_s=now_s,latest_stop_s=now_s+p.total_s+config.reaction_s,
        motion_assumption='hypothetical_rest_to_rest_straight_segment',
        reaction_assumption='hold_trigger_speed_then_constant_brake',physical_bounds_validated=False)
    return volume,envelope


def evaluate_motion(inspector,candidate,config,*,now_s,clock_id,error_bound_m):
    if type(inspector) is not DepthVolumeInspector: raise ValueError('bound depth inspector required')
    if type(candidate) is not RestMotionCandidate: raise ValueError('validated motion candidate required')
    if (candidate.world_frame!=inspector.world_frame or candidate.reference_fingerprint!=inspector.reference_fingerprint):
        raise ValueError('candidate reference differs from depth')
    volume,envelope=stopping_volume(candidate,config,now_s=now_s,error_bound_m=error_bound_m)
    geometry=inspector.inspect(volume,now_s=now_s,clock_id=clock_id)
    remaining=inspector.valid_until_s-now_s
    expires=envelope['latest_stop_s']>inspector.valid_until_s+1e-9
    reasons=list(geometry['reasons'])
    if expires: reasons.append('OBSERVATION_EXPIRES_BEFORE_STOP')
    reasons.append('PHYSICAL_MOTION_BOUNDS_UNVALIDATED')
    return dict(envelope=envelope,geometry=geometry,remaining_evidence_s=remaining,
        stopping_horizon_s=envelope['latest_stop_s']-now_s,expires_before_stop=expires,
        reasons=reasons,status='HOLD',selected_for_execution=False,flight_authorized=False,
        navigation_map_update_allowed=False)


def planning_comparison(inspector,*,origin,step_m,start,goal,config,now_s,clock_id,error_bound_m):
    """Compare a 3x3 declared-free A* baseline with its first-step depth checks.

    This is a local planning diagnostic, not an inferred free map. All four
    reachable first-step alternatives are checked, including a path away from
    the goal. No movement is selected without coverage and control evidence.
    """
    if type(inspector) is not DepthVolumeInspector: raise ValueError('bound depth inspector required')
    if type(origin) is not tuple or len(origin)!=3 or not all(number(v) for v in origin) or not number(step_m) or not .01<=step_m<=2:
        raise ValueError('bounded research planning coordinates required')
    for cell in (start,goal):
        if type(cell) is not tuple or len(cell)!=2 or any(type(v) is not int or not 0<=v<3 for v in cell):
            raise ValueError('3x3 planning cells required')
    if start==goal: raise ValueError('comparison requires a movement goal')
    def point(cell): return (origin[0]+cell[0]*step_m,origin[1],origin[2]+cell[1]*step_m)
    baseline=astar(3,3,start,goal,set(),margin=0)
    alternatives=[]
    for x,z in ((start[0]+1,start[1]),(start[0],start[1]+1),(start[0]-1,start[1]),(start[0],start[1]-1)):
        if not 0<=x<3 or not 0<=z<3: continue
        tail=astar(3,3,(x,z),goal,{start},margin=0)
        if not tail: continue
        candidate=RestMotionCandidate(f'到格({x},{z})',inspector.world_frame,inspector.reference_fingerprint,point(start),point((x,z)))
        result=evaluate_motion(inspector,candidate,config,now_s=now_s,clock_id=clock_id,error_bound_m=error_bound_m)
        alternatives.append(dict(next_cell=[x,z],baseline_remaining_steps=len(tail)-1,result=result))
    alternatives.sort(key=lambda row:(row['baseline_remaining_steps'],row['next_cell']))
    return dict(grid_assumption='counterfactual_all_cells_free_for_point_A_star',origin=list(origin),step_m=step_m,
        plane='reference_XZ_not_gravity_horizontal',baseline_route=baseline,alternatives=alternatives,
        selected_route=[],status='HOLD_NO_EXECUTABLE_CANDIDATE',flight_authorized=False,
        note='An A* route in an assumed-free map is not measured free-space evidence.')
