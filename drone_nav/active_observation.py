"""来源：本项目原创。已获条件支持区域内的小幅物理移动与新相机观测。"""
from dataclasses import dataclass, asdict
from itertools import product
from math import sqrt, atan2, cos, sin, pi

from .detection_bridge import number
from .pinhole import Intrinsics, Pose, PerspectiveFrame
from .physical_vehicle import rotate_vector
from .sampling_motion import SamplingView
from .refined_sampling import RefinedMissionVehicle
from .guarded_mission import GuardedMissionVehicle


@dataclass(frozen=True)
class ProbeConfig:
    enabled: bool = True
    offset_m: float = .12
    outbound_s: float = .8
    opposite_s: float = .8
    return_s: float = 2.0
    frame_interval_s: float = .04
    processing_s: float = .02

    def __post_init__(self):
        if type(self.enabled) is not bool or any(not number(v) or v<=0 for k,v in asdict(self).items() if k!='enabled'):
            raise ValueError('invalid active observation configuration')
        if self.offset_m>.12 or self.duration>4. or self.processing_s>=self.frame_interval_s:
            raise ValueError('active observation exceeds development bounds')

    @property
    def duration(self):return self.outbound_s+self.opposite_s+self.return_s


def aim_at_box(state,low,high,k):
    """根据待观察盒的投影选转台朝向；只读已知坐标，不读取场景真值。"""
    origin=tuple(state['position']);corners=list(product(*zip(low,high)))
    vectors=[tuple(a-b for a,b in zip(p,origin)) for p in corners]
    directions=[tuple(a/max(1e-12,sqrt(sum(v*v for v in d))) for a in d) for d in vectors]
    direction=tuple(sum(d[i] for d in directions) for i in range(3))
    yaw=atan2(direction[1],direction[0]);pitch=atan2(direction[2],sqrt(direction[0]**2+direction[1]**2))
    best=None;best_score=-float('inf')
    for dy,dp in product(range(-30,31,10),range(-20,21,10)):
        y=yaw+dy*pi/180;p=pitch+dp*pi/180
        target=(origin[0]+cos(p)*cos(y),origin[1]+cos(p)*sin(y),origin[2]+sin(p))
        base=Pose.look_at(origin,target);q=state['quaternion']
        actual=Pose(origin,rotate_vector(base.right,q),rotate_vector(base.down,q),rotate_vector(base.forward,q))
        projected=[actual.project(c,k) for c in corners]
        score=-float('inf') if any(x is None for x in projected) else min(
            min(x+.5,k.width-.5-x,y+.5,k.height-.5-y) for x,y,z in projected)
        if best is None or score>best_score:best=target;best_score=score
    return best


class ActiveObservationMixin:
    def __init__(self,*args,probe_config=None,**kwargs):
        config=ProbeConfig() if probe_config is None else probe_config
        if not isinstance(config,ProbeConfig):raise ValueError('invalid probe configuration')
        super().__init__(*args,**kwargs)
        self.probe_config=config;self.probes=[]

    def scan(self,tick,aim):
        self._navigation_aim=tuple(aim)
        self.retire_expired_views()
        if self.probes and self.probes[-1]['status']=='RETURN_CONFIRMED':
            # Fresh moving views already cover additional directions. Keep the
            # original eight new navigation images and their real acquisition age.
            return GuardedMissionVehicle.scan(self,tick,aim)
        return super().scan(tick,aim)

    def retire_expired_views(self):
        # Remove only frames whose original lifetime has already elapsed. Keep
        # coarse occupied cells and original map stamps; never refresh a source.
        now=self.state()['time_s']
        self.sampling_history._entries=[entry for entry in self.sampling_history._entries
            if entry[0].captured_at_s+self.budget.free_ttl_s>=now]

    def move(self,target_cell):
        previous_cell=self.confirmed_cell
        receipt=super().move(target_cell)
        if (not receipt['completed'] or not self.probe_config.enabled or
                tuple(target_cell)==getattr(self,'_navigation_aim',tuple(target_cell))):
            return receipt
        try:
            self.observe_nearfield()
        except (ValueError,TypeError,KeyError,RuntimeError,OverflowError) as exc:
            self.confirmed_cell=previous_cell
            if self.probes and self.probes[-1]['status']=='OBSERVING':
                self.probes[-1].update(status='PROBE_PROTOCOL_HOLD',error=str(exc),actual_final=self.state())
            return self.probe_receipt(receipt,target_cell,False,'ACTIVE_OBSERVATION:'+str(exc))
        if self.probes[-1]['status']!='RETURN_CONFIRMED':
            self.confirmed_cell=previous_cell
            return self.probe_receipt(receipt,target_cell,False,'ACTIVE_OBSERVATION:'+self.probes[-1]['status'])
        return self.probe_receipt(receipt,target_cell,True,'ARRIVED_AFTER_ACTIVE_OBSERVATION')

    def probe_receipt(self,receipt,target_cell,completed,reason):
        final=self.state();center=(target_cell[0]+.5,target_cell[1]+.5,3.5)
        probe=self.probes[-1]
        return dict(receipt,end=final,arrival_receipt=receipt,reason=reason,completed=completed,
                    duration_s=final['time_s']-receipt['start']['time_s'],
                    position_error_m=sqrt(sum((a-b)**2 for a,b in zip(final['position'],center))),
                    speed_mps=sqrt(sum(v*v for v in final['velocity'])),
                    max_speed_mps=max(receipt['max_speed_mps'],probe.get('max_speed_mps',0.)),
                    max_height_error_m=max(receipt['max_height_error_m'],probe.get('max_height_error_m',0.)),
                    max_tracking_error_m=max(receipt['max_tracking_error_m'],probe.get('max_center_error_m',0.)),
                    stop_confirmed=completed,stable_duration_s=probe.get('stable_duration_s',0.))

    def observe_nearfield(self):
        """只在前次实际移动成功后使用当时的完整覆盖，不创建或刷新空闲声明。"""
        config=self.probe_config;before=self.state()
        record=dict(actual_start=before,status='NOT_AUTHORIZED',frames=[],config=asdict(config))
        self.probes.append(record)
        if (not self.move_checks or not self.move_checks[-1]['receipt']
                or not self.move_checks[-1]['receipt']['completed'] or self.motor_latched_off):
            record['actual_final']=self.state();return
        guard=self.move_checks[-1]['final_guard']
        if not guard['allowed']:
            record['actual_final']=self.state();return
        cells=guard['required_cells']
        low=tuple(min(c[i] for c in cells) for i in range(3))
        high=tuple(max(c[i]+1 for c in cells) for i in range(3))
        # The swept center and body must stay in the same fully covered rectangle.
        expected=set(product(*(range(int(a),int(b)) for a,b in zip(low,high))))
        if {tuple(c) for c in cells}!=expected:raise ValueError('nonrectangular probe support')
        stop=min(guard['baseline']['latest_stop_s'],
                 *(p['source']['valid_until_s'] for r in guard['sampling'] for p in r['patches']))
        center=(self.confirmed_cell[0]+.5,self.confirmed_cell[1]+.5,3.5)
        record.update(support_low=low,support_high=high,support_until_s=stop,target=center)
        if before['time_s']+config.duration>stop+1e-9:
            record.update(status='INSUFFICIENT_TIME',actual_final=self.state());return
        dt=self.physics.dt
        for value in (config.duration,config.frame_interval_s,config.processing_s):
            if abs(round(value/dt)*dt-value)>1e-9:raise ValueError('probe duration must fit physics steps')
        radius=self.budget.body_radius_m+self.budget.tracking_margin_m
        def contained(position):return all(a+radius<=x<=b-radius for x,a,b in zip(position,low,high))
        if not all(contained((center[0]+sign*config.offset_m,center[1]+sign*config.offset_m,center[2])) for sign in (-1,0,1)):
            record.update(status='OUTSIDE_SUPPORTED_REGION',actual_final=self.state());return
        k=Intrinsics(80,60,12,12,39.5,29.5)
        points=[(-.125,.125,z) for z in (-.125,.125)]+[(-.125,-.125,z) for z in (-.125,.125)]+[
            (.125,y,z) for y,z in product((-.125,.125),repeat=2)]
        interval=round(config.frame_interval_s/dt)
        sequence=self.sampling_history.grid.last_tick+1;pending=[];stable=0;captured=0
        record.update(status='OBSERVING',max_speed_mps=0.,max_center_error_m=0.,max_height_error_m=0.)
        self.hold_target=center
        for step in range(round(config.duration/dt)):
            elapsed=step*dt
            sign=1 if elapsed<config.outbound_s else -1 if elapsed<config.outbound_s+config.opposite_s else 0
            target=(center[0]+sign*config.offset_m,center[1]+sign*config.offset_m,center[2])
            state=self.state()
            if step%interval==0:
                box_center=tuple(a+b for a,b in zip(center,points[captured%len(points)]))
                point=aim_at_box(state,tuple(v-.125 for v in box_center),tuple(v+.125 for v in box_center),k)
                frame=self.capture(k,sequence,lambda xyz,point=point:Pose.look_at(xyz,point))
                base=Pose.look_at(tuple(state['position']),point);q=state['quaternion']
                expected=Pose(tuple(state['position']),rotate_vector(base.right,q),
                              rotate_vector(base.down,q),rotate_vector(base.forward,q))
                if not isinstance(frame,PerspectiveFrame) or frame.pose!=expected or frame.intrinsics!=k or frame.tick!=sequence:
                    raise ValueError('active camera pose, calibration or sequence mismatch')
                frame.validate()
                view=SamplingView(frame,f'probe-{sequence}','camera','physics-seconds',state['time_s'],state['time_s']+config.processing_s)
                pending.append(view);sequence+=1;captured+=1
                record['frames'].append(dict(frame_id=view.frame_id,captured_at_s=view.captured_at_s,available_at_s=view.available_at_s,
                                             capture_state=state,aim=point))
            state=self._step(target)
            if not contained(state['position']) or state['time_s']>stop+1e-9:
                record.update(status='PROBE_ENVELOPE_BREACH',actual_final=state)
                self.hold_target=tuple(state['position'])
                raise ValueError('active observation exceeded previous support')
            while pending and pending[0].available_at_s<=state['time_s']+1e-9:
                view=pending.pop(0)
                self.retire_expired_views()
                # Avoid a tiny floating-point early availability at the consumer.
                self.sampling_history.add(view,now_s=max(state['time_s'],view.available_at_s))
            speed=sqrt(sum(x*x for x in state['velocity']))
            error=sqrt(sum((x-y)**2 for x,y in zip(state['position'],center)))
            record['max_speed_mps']=max(record['max_speed_mps'],speed)
            record['max_center_error_m']=max(record['max_center_error_m'],error)
            record['max_height_error_m']=max(record['max_height_error_m'],abs(state['position'][2]-3.5))
            stable=stable+1 if sign==0 and speed<=self.budget.speed_tolerance_mps and error<=self.budget.position_tolerance_m else 0
        record.update(actual_final=self.state(),stable_duration_s=stable*dt,pending_frames=len(pending),
                      status='RETURN_CONFIRMED' if stable*dt>=self.budget.stable_s and not pending else 'RETURN_UNCONFIRMED')
        if record['status']!='RETURN_CONFIRMED':
            raise ValueError('active observation return not confirmed')


class ActiveObservationVehicle(ActiveObservationMixin,RefinedMissionVehicle):
    pass
