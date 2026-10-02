"""来源：本项目原创。短周期运动许可与独立超时停止的运动学研究模型。"""
from dataclasses import dataclass,asdict
from .detection_bridge import identifier,number


@dataclass(frozen=True)
class RollingConfig:
    dt_s: float=.02
    acceleration_mps2: float=1.
    brake_mps2: float=1.
    max_speed_mps: float=.2
    reaction_s: float=.2
    body_radius_m: float=.25
    position_error_m: float=.02
    margin_m: float=.05
    ttl_s: float=.5

    def __post_init__(self):
        if any(not number(v) or v<=0 for v in (self.dt_s,self.acceleration_mps2,self.brake_mps2,self.max_speed_mps,self.body_radius_m,self.ttl_s)):
            raise ValueError('positive finite motion and lifetime parameters required')
        if any(not number(v) or v<0 for v in (self.reaction_s,self.position_error_m,self.margin_m)):
            raise ValueError('nonnegative reaction and margins required')
        if self.dt_s>.1 or self.max_speed_mps>2 or self.ttl_s>1:
            raise ValueError('research controller bounds exceeded')

    @property
    def padding_m(self): return self.body_radius_m+2*self.position_error_m+self.margin_m


@dataclass(frozen=True)
class IntervalObservation:
    frame_id: str
    camera_id: str
    clock_id: str
    captured_at_s: float
    available_at_s: float
    lower_m: float
    upper_m: float
    source: str
    coverage_model: str='ideal_full_corridor_interval'

    def __post_init__(self):
        if not all(identifier(v) for v in (self.frame_id,self.camera_id,self.clock_id,self.source,self.coverage_model)):
            raise ValueError('observation identity and source required')
        if (any(not number(v) for v in (self.captured_at_s,self.available_at_s,self.lower_m,self.upper_m))
                or self.captured_at_s<0 or self.available_at_s<self.captured_at_s or self.lower_m>=self.upper_m):
            raise ValueError('invalid observation time or interval')


def advance(x,v,a,dt):
    """Exact constant-acceleration integration, without reversing after a stop."""
    duration=min(dt,v/-a) if a<0 else dt
    speed=max(0.,v+a*duration)
    return x+v*duration+.5*a*duration*duration,0. if speed<1e-12 else speed


def envelope(x,v,a,now,config):
    x1,v1=advance(x,v,a,config.dt_s)
    # For acceleration/coast the endpoint is worst; for full braking the start
    # is worst. The guard only emits these actions, not arbitrary partial brakes.
    stop0=x+v*config.reaction_s+v*v/(2*config.brake_mps2)
    stop1=x1+v1*config.reaction_s+v1*v1/(2*config.brake_mps2)
    t0=now+config.reaction_s+v/config.brake_mps2
    t1=now+config.dt_s+config.reaction_s+v1/config.brake_mps2
    return dict(next_x_m=x1,next_speed_mps=v1,lower_m=x-config.padding_m,
        upper_m=max(stop0,stop1)+config.padding_m,latest_stop_s=max(t0,t1))


@dataclass(frozen=True)
class MotionTicket:
    at_s: float
    x_m: float
    speed_mps: float
    acceleration_mps2: float
    observation: IntervalObservation


class RollingGuard:
    def __init__(self,config,*,camera_id,clock_id):
        if type(config) is not RollingConfig or not all(identifier(v) for v in (camera_id,clock_id)):
            raise ValueError('validated rolling configuration/identities required')
        self.config=config; self.camera_id=camera_id; self.clock_id=clock_id
        self.latest=None; self.fault=None; self._now=-1.

    def _time(self,now):
        if not number(now) or now<0 or now<self._now-1e-9: raise ValueError('monotonic controller time required')

    def receive(self,observation,*,now_s):
        self._time(now_s)
        if type(observation) is not IntervalObservation: raise ValueError('validated interval observation required')
        reason=None
        if self.fault: reason='STREAM_FAULT_LATCHED'
        elif observation.camera_id!=self.camera_id: reason='CAMERA_MISMATCH'
        elif observation.clock_id!=self.clock_id: reason='CLOCK_MISMATCH'
        elif observation.coverage_model!='ideal_full_corridor_interval': reason='COVERAGE_UNPROVEN'
        elif observation.available_at_s>now_s+1e-9: reason='FRAME_NOT_AVAILABLE'
        elif now_s>observation.captured_at_s+self.config.ttl_s+1e-9: reason='FRAME_EXPIRED'
        elif self.latest and observation.frame_id==self.latest.frame_id: reason='REPEATED_FRAME'
        elif self.latest and observation.captured_at_s<=self.latest.captured_at_s: reason='OUT_OF_ORDER_FRAME'
        if reason:
            # Rejected frames never replace/refresh the last committed evidence.
            self.fault=self.fault or reason
        else: self.latest=observation
        self._now=now_s
        return dict(accepted=reason is None,reason=reason or 'ACCEPTED',fault=self.fault)

    def choose(self,*,now_s,x_m,speed_mps,goal_m):
        self._time(now_s)
        if any(not number(v) for v in (x_m,speed_mps,goal_m)) or not 0<=speed_mps<=self.config.max_speed_mps+1e-8:
            raise ValueError('finite forward-motion state required')
        self._now=now_s; c=self.config
        if self.fault or self.latest is None: return None,dict(reason=self.fault or 'NO_OBSERVATION',checks=[])
        if speed_mps<=1e-10:
            if abs(goal_m-x_m)<=.005: return None,dict(reason='GOAL_STOPPED',checks=[])
            if x_m>goal_m: return None,dict(reason='GOAL_PASSED_HOLD',checks=[])
        acceleration=min(c.acceleration_mps2,max(0.,(c.max_speed_mps-speed_mps)/c.dt_s))
        choices=list(dict.fromkeys((acceleration,0.,-c.brake_mps2))); checks=[]
        for a in choices:
            if speed_mps<=1e-10 and a<=0: continue
            bound=envelope(x_m,speed_mps,a,now_s,c); reasons=[]
            nominal_stop=bound['next_x_m']+bound['next_speed_mps']**2/(2*c.brake_mps2)
            if a>=0 and nominal_stop>goal_m+1e-9: reasons.append('GOAL_BRAKING_REQUIRED')
            if bound['lower_m']<self.latest.lower_m-1e-9 or bound['upper_m']>self.latest.upper_m+1e-9:
                reasons.append('STOP_OUTSIDE_OBSERVED_INTERVAL')
            if bound['latest_stop_s']>self.latest.captured_at_s+c.ttl_s+1e-9:
                reasons.append('EXPIRES_BEFORE_WATCHDOG_STOP')
            checks.append(dict(acceleration_mps2=a,bound=bound,reasons=reasons))
            if not reasons:
                return MotionTicket(now_s,x_m,speed_mps,a,self.latest),dict(reason='MODEL_STEP_WITH_FALLBACK',checks=checks,
                    flight_authorized=False)
        return None,dict(reason='NO_SUPPORTED_NEXT_STEP',checks=checks)


class WatchdogExecutor:
    """Independent model executor: missing next ticket starts the last fallback.

    Once fallback begins it cannot be cancelled by a later frame/ticket. This
    does not implement a hardware watchdog or a physical flight controller.
    """
    def __init__(self,config,*,camera_id,clock_id,x_m=0.):
        if type(config) is not RollingConfig or not number(x_m) or not all(identifier(v) for v in (camera_id,clock_id)):
            raise ValueError('validated executor state and stream identity required')
        self.config=config; self.now_s=0.; self.x_m=x_m; self.speed_mps=0.
        self.camera_id=camera_id; self.clock_id=clock_id
        self.committed=None; self.fallback_at_s=None; self.mode='WAITING'

    def step(self,ticket=None,*,brake_efficiency=1.):
        c=self.config
        if not number(brake_efficiency) or not 0<brake_efficiency<=1: raise ValueError('invalid injected brake efficiency')
        before=dict(now_s=self.now_s,x_m=self.x_m,speed_mps=self.speed_mps)
        if ticket is not None and self.fallback_at_s is None:
            if type(ticket) is not MotionTicket or type(ticket.observation) is not IntervalObservation:
                raise ValueError('validated short-lived ticket required')
            if (any(not number(v) for v in (ticket.at_s,ticket.x_m,ticket.speed_mps))
                    or abs(ticket.at_s-self.now_s)>1e-9 or abs(ticket.x_m-self.x_m)>1e-9
                    or abs(ticket.speed_mps-self.speed_mps)>1e-9): raise ValueError('ticket does not match current executor state')
            a=ticket.acceleration_mps2
            if not number(a) or not (0<=a<=c.acceleration_mps2 or a==-c.brake_mps2): raise ValueError('unsupported control action')
            bound=envelope(self.x_m,self.speed_mps,a,self.now_s,c); obs=ticket.observation
            if (obs.camera_id!=self.camera_id or obs.clock_id!=self.clock_id
                    or obs.coverage_model!='ideal_full_corridor_interval' or obs.available_at_s>self.now_s+1e-9
                    or bound['latest_stop_s']>obs.captured_at_s+c.ttl_s+1e-9
                    or bound['lower_m']<obs.lower_m-1e-9 or bound['upper_m']>obs.upper_m+1e-9
                    or bound['next_speed_mps']>c.max_speed_mps+1e-9):
                raise ValueError('ticket lacks a valid model fallback')
            self.committed=dict(ticket=asdict(ticket),bound=bound)
            effective=a*brake_efficiency if a<0 else a
            self.x_m,self.speed_mps=advance(self.x_m,self.speed_mps,effective,c.dt_s)
            self.mode='CONTROL'
            parts=[dict(duration_s=c.dt_s,acceleration_mps2=effective)]
        elif self.speed_mps>1e-10:
            if self.committed is None: raise ValueError('moving executor has no committed fallback')
            if self.fallback_at_s is None: self.fallback_at_s=self.now_s
            hold=min(c.dt_s,max(0.,self.fallback_at_s+c.reaction_s-self.now_s))
            self.x_m,self.speed_mps=advance(self.x_m,self.speed_mps,0.,hold)
            self.x_m,self.speed_mps=advance(self.x_m,self.speed_mps,-c.brake_mps2*brake_efficiency,c.dt_s-hold)
            self.mode='FALLBACK' if self.speed_mps>1e-10 else 'STOPPED'
            parts=[dict(duration_s=hold,acceleration_mps2=0.),dict(duration_s=c.dt_s-hold,acceleration_mps2=-c.brake_mps2*brake_efficiency)]
        else:
            self.mode='STOPPED' if self.committed else 'WAITING'
            parts=[dict(duration_s=c.dt_s,acceleration_mps2=0.)]
        self.now_s=round(self.now_s+c.dt_s,10)
        return dict(before=before,after=dict(now_s=self.now_s,x_m=self.x_m,speed_mps=self.speed_mps),
            mode=self.mode,parts=parts,fallback_at_s=self.fallback_at_s,committed=self.committed,
            flight_authorized=False,model_only=True)
