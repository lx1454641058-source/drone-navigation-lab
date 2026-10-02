"""来源：本项目原创。视觉授权的低速下降、有限视野证据和物理接地确认。"""

from dataclasses import asdict,dataclass
from math import acos,degrees,hypot,isfinite,sqrt

from .perspective_landing import LandingConfig,inspect_landing,landing_preview
from .pinhole import Intrinsics


@dataclass(frozen=True)
class DescentConfig:
    rate_mps: float=.15
    max_duration_s: float=40.0
    observation_period_s: float=.5
    processing_s: float=.1
    corridor_radius_m: float=.60
    certificate_ttl_s: float=8.0
    horizontal_tolerance_m: float=.10
    max_speed_mps: float=.40
    max_tilt_deg: float=8.0
    contact_gap_m: float=.002
    max_penetration_m: float=.005
    contact_speed_mps: float=.03
    contact_stable_s: float=.30
    disarmed_settle_s: float=1.0
    abort_hold_s: float=2.0

    def __post_init__(self):
        if any(type(v) not in (int,float) or not isfinite(v) or v<=0 for v in asdict(self).values()):
            raise ValueError('descent limits must be finite positive numbers')
        if (self.rate_mps>self.max_speed_mps or self.max_speed_mps>.6 or self.max_duration_s>90
                or self.processing_s>=self.observation_period_s or self.observation_period_s>1
                or self.corridor_radius_m<.35+self.horizontal_tolerance_m or self.corridor_radius_m>2
                or self.max_tilt_deg>15 or self.contact_gap_m>.005 or self.max_penetration_m>.01):
            raise ValueError('inconsistent descent limits')
        if self.disarmed_settle_s<self.contact_stable_s:
            raise ValueError('disarmed verification must cover the stability duration')
        # 所有模拟计时必须能落在固定物理步上，避免 round 暗中改变实验参数。
        for name in ('max_duration_s','observation_period_s','processing_s','contact_stable_s','disarmed_settle_s','abort_hold_s'):
            value=getattr(self,name)/.002
            if abs(value-round(value))>1e-8:raise ValueError('duration must align with the 0.002 s physical step')


def tilt_deg(state):
    q=state['quaternion']
    return degrees(acos(max(-1,min(1,1-2*(q[1]**2+q[2]**2)))))


def envelope_reason(state,xy,c):
    if hypot(state['position'][0]-xy[0],state['position'][1]-xy[1])>c.horizontal_tolerance_m:
        return 'LATERAL_DRIFT'
    if sqrt(sum(v*v for v in state['velocity']))>c.max_speed_mps:return 'EXCESSIVE_SPEED'
    if tilt_deg(state)>c.max_tilt_deg:return 'EXCESSIVE_TILT'
    return None


def contact_candidate(state,gap,xy,c):
    return (gap is not None and isfinite(gap) and -c.max_penetration_m<=gap<=c.contact_gap_m
            and abs(state['position'][2]-.06)<=.012
            and sqrt(sum(v*v for v in state['velocity']))<=c.contact_speed_mps
            and sqrt(sum(v*v for v in state['angular_velocity']))<=.1
            and tilt_deg(state)<=2 and envelope_reason(state,xy,c) is None)


def descend_after_navigation(vehicle,model,navigation,*,config=None,previews=True):
    c=config or DescentConfig();start=vehicle.state();xy=tuple(v+.5 for v in navigation['goal'])
    result=dict(config=asdict(c),started_at_s=start['time_s'],ended_at_s=start['time_s'],
                status='NOT_AUTHORIZED',reason='巡航或末端图像未授权降落',events=[],trace=[],
                disarmed_at_s=None,stop_confirmed=False,final_state=start,
                limitation='静态场景假设；低空视野缩小时仅局部重观察，完整通道依赖有限时效旧证据。几何间距探针使用仿真真值，非真实接触传感器。')
    if (navigation['terminal_state']!='READY_TO_LAND' or not navigation.get('landing',{}).get('accepted',False)
            or navigation['model_digest']!=model.digest or navigation['actual_final']!=start):
        return result
    if (hypot(start['position'][0]-xy[0],start['position'][1]-xy[1])>.03
            or abs(start['position'][2]-3.5)>.03 or sqrt(sum(v*v for v in start['velocity']))>.03):
        result['reason']='物理到达条件已不满足';return result
    vehicle.descent_start_s=start['time_s']
    k=Intrinsics(80,60,22,22,39.5,29.5)
    events,trace=result['events'],result['trace'];certificate_at=None;last_gap=None
    base_tick=navigation['trace'][-1]['tick']+1;stable=0;phase='DESCEND'
    target=(xy[0],xy[1],start['position'][2]);max_steps=round(c.max_duration_s/.002)
    next_observation=0;reason=None;contact_ready=False;step0=vehicle.steps

    def record(current_phase,force=False):
        s=vehicle.state()
        if force or (vehicle.steps-step0)%50==0:
            trace.append(dict(state=s,phase=current_phase,target_z_m=target[2],gap_m=last_gap,
                              last_event=len(events)-1,certificate_age_s=None if certificate_at is None else s['time_s']-certificate_at))

    def advance(command,off=False):
        nonlocal last_gap
        state=vehicle.apply_motors([0.0]*4) if off else vehicle._step(command)
        if (vehicle.steps-step0)%10==0:last_gap=vehicle.contact_gap()
        record(phase)
        return state

    while vehicle.steps-step0<max_steps:
        current=vehicle.state();elapsed=current['time_s']-start['time_s']
        reason=envelope_reason(current,xy,c)
        if reason:break
        if vehicle.steps-step0>=next_observation:
            # 处理图像也会推进物理：旧证据必须覆盖整个处理窗口，不能等处理完再发现过期。
            if certificate_at is not None and current['time_s']+c.processing_s-certificate_at>c.certificate_ttl_s+1e-9:
                reason='CORRIDOR_EVIDENCE_EXPIRED';break
            capture_state=current;tick=base_tick+len(events)
            # 在机体接近地面时，当前相机确实看不到整个 .60 m 圆盘；不伪造全覆盖。
            visible_radius=min(c.corridor_radius_m,max(.025,(current['position'][2]-.02)*.9*k.height/(2*k.fy)-hypot(current['position'][0]-xy[0],current['position'][1]-xy[1])))
            full=visible_radius>=c.corridor_radius_m-1e-9
            event=dict(tick=tick,captured_at_s=current['time_s'],decided_at_s=None,full_corridor=full,
                       visible_radius_m=visible_radius,evidence=None,images=None,error=None)
            try:
                frame=vehicle.capture(k,tick)
                if frame.intrinsics!=k:raise ValueError('descent camera calibration mismatch')
                # 处理延迟期间仍执行上一个目标，避免把图像计算时间当作物理暂停。
                for _ in range(round(c.processing_s/.002)):
                    state=advance(target)
                    failure=envelope_reason(state,xy,c)
                    if failure:raise ValueError('motion envelope during processing: '+failure)
                evidence,labels=inspect_landing(frame,model,(*xy,0),expected_tick=tick,
                    expected_position=capture_state['position'],config=LandingConfig(radius_m=visible_radius,
                    max_roughness_m=.008,max_residual_m=.015,elevation_tolerance_m=.02))
                if evidence['accepted']:
                    evidence['reason']='本帧图像与几何检查通过；接地状态须另行确认'
                event['evidence']=evidence
                event['images']=landing_preview(frame,evidence,labels) if previews else None
                if not evidence['accepted']:reason='VISION_REJECTED'
                elif full:certificate_at=event['captured_at_s']
            except (ValueError,TypeError,OverflowError) as exc:
                event['error']=str(exc);reason='VISION_PROTOCOL_ERROR'
            event['decided_at_s']=vehicle.state()['time_s'];events.append(event)
            next_observation=vehicle.steps-step0+round((c.observation_period_s-c.processing_s)/.002)
            record(phase,True)
            if reason:break
        current=vehicle.state()
        if certificate_at is None or current['time_s']+.002-certificate_at>c.certificate_ttl_s+1e-9:
            reason='CORRIDOR_EVIDENCE_EXPIRED';break
        phase='DESCEND' if events[-1]['full_corridor'] else 'TERMINAL_LOCAL_VIEW'
        # 低速目标斜坡；真实机体由控制器和引擎追踪，绝不直接写入位置。
        target=(*xy,max(.055,start['position'][2]-c.rate_mps*(current['time_s']-start['time_s'])))
        state=advance(target)
        if (vehicle.steps-step0)%10==0:
            if last_gap is None:
                reason='CONTACT_SENSOR_INVALID';break
            stable=stable+10 if contact_candidate(state,last_gap,xy,c) else 0
            if stable*.002>=c.contact_stable_s:
                contact_ready=True;break
    if contact_ready:
        phase='DISARM_VERIFY';result['disarmed_at_s']=vehicle.state()['time_s'];stable_after=True
        record(phase,True)
        for _ in range(round(c.disarmed_settle_s/.002)):
            state=advance(target,off=True)
            # 容许停桨瞬间的微小沉降，最后 .30 秒必须连续稳定接地。
            if _>=round((c.disarmed_settle_s-c.contact_stable_s)/.002) and (vehicle.steps-step0)%10==0:
                stable_after=stable_after and contact_candidate(state,last_gap,xy,c)
        result['status']='LANDED' if stable_after else 'TOUCHDOWN_UNCONFIRMED'
        result['reason']='停桨后持续满足位置、速度和几何接地条件' if stable_after else '停桨后接地稳定条件不满足'
        result['stop_confirmed']=stable_after
    else:
        phase='ABORT_HOLD';reason=reason or 'DESCENT_TIMEOUT';target=tuple(vehicle.state()['position'])
        vehicle.hold_target=target;stable=0;record(phase,True)
        for _ in range(round(c.abort_hold_s/.002)):
            state=advance(target)
            speed=sqrt(sum(v*v for v in state['velocity']))
            stable=stable+1 if speed<=c.contact_speed_mps and envelope_reason(state,xy,c) is None else 0
        stopped=stable*.002>=c.contact_stable_s
        result['status']='ABORTED_HOLD' if stopped else 'ABORT_STOP_UNCONFIRMED'
        result['reason']=reason;result['stop_confirmed']=stopped
    record(phase,True)
    result.update(ended_at_s=vehicle.state()['time_s'],final_state=vehicle.state())
    return result
