"""来源：本项目原创。几何扫描后感知、实际移动推力提交前逐步检查。"""
from dataclasses import dataclass
from math import sqrt

from .detection_bridge import number,identifier
from .pinhole import finite_vector,Intrinsics
from .physical_vehicle import PhysicalVehicle
from .guarded_mission import GuardedMissionVehicle
from .physical_rescan import rescan_and_move
from .mission_supervisor import MotorInterlockError


@dataclass(frozen=True)
class VisualMotionLease:
    """An empty-detection veto clearance, not proof that the scene is empty."""
    frame_id: str
    target: tuple
    captured_at_s: float
    available_at_s: float
    geometry_valid_until_s: float
    clock_id: str='physics-seconds'

    def __post_init__(self):
        if (not identifier(self.frame_id) or type(self.target) is not tuple or not finite_vector(self.target,3)
            or self.clock_id!='physics-seconds' or any(not number(v) or v<0 for v in
            (self.captured_at_s,self.available_at_s,self.geometry_valid_until_s))
            or not self.captured_at_s<=self.available_at_s<=self.captured_at_s+.5+1e-9):
            raise ValueError('explicit fresh same-clock motion observation required')


def submission_check(lease,*,target,now_s,dt,clock_id,command_state,current_state,
                     move_started_at_s,max_move_s=8.):
    """Check the next physical interval, not just its starting instant."""
    reason='VISUAL_STEP_CURRENT'
    if type(lease) is not VisualMotionLease:
        reason='MISSING_VISUAL_LEASE'
    elif clock_id!=lease.clock_id:
        reason='ACTUATOR_CLOCK_MISMATCH'
    elif (not number(now_s) or not number(dt) or not 0<dt<=.02 or not number(move_started_at_s)
          or not number(max_move_s) or max_move_s<=0 or now_s<move_started_at_s-1e-9
          or now_s<lease.available_at_s-1e-9):
        reason='ACTUATOR_TIME_INVALID'
    elif not finite_vector(target,3) or tuple(target)!=lease.target:
        reason='ACTUATOR_TARGET_MISMATCH'
    elif (not isinstance(current_state,dict) or current_state.get('time_s')!=now_s
          or not isinstance(command_state,dict) or not number(command_state.get('time_s'))):
        reason='ACTUATOR_STATE_INVALID'
    elif now_s+dt>lease.captured_at_s+.5+1e-9:
        reason='VISUAL_EXPIRED_AT_ACTUATOR'
    elif command_state!=current_state:
        reason='CONTROL_STATE_CHANGED_BEFORE_SUBMISSION'
    elif now_s+dt>move_started_at_s+max_move_s+1e-9:
        reason='MOTION_DURATION_EXPIRED'
    elif move_started_at_s+max_move_s+.5>lease.geometry_valid_until_s+1e-9:
        reason='GEOMETRY_EXPIRES_BEFORE_STOP'
    return dict(allowed=reason=='VISUAL_STEP_CURRENT',reason=reason,checked_at_s=now_s,
                step_end_s=now_s+dt if number(now_s) and number(dt) else None,
                valid_until_s=lease.captured_at_s+.5 if type(lease) is VisualMotionLease else None,
                flight_authorized=False)


class MotionVeto(RuntimeError):
    pass


def rejected_receipt(vehicle,before,target,reason):
    final=vehicle.state()
    return dict(completed=False,reason=reason,start=before,end=final,target=tuple(target),
        duration_s=final['time_s']-before['time_s'],speed_mps=sqrt(sum(v*v for v in final['velocity'])),
        stop_confirmed=False)


class _PostScanExecutor:
    def __init__(self,vehicle):self.vehicle=vehicle
    def __getattr__(self,name):return getattr(self.vehicle,name)
    @property
    def hold_target(self):return self.vehicle.hold_target
    @hold_target.setter
    def hold_target(self,value):self.vehicle.hold_target=value
    def move(self,target):return self.vehicle.move_after_scan(target)


class ActuatorVisionVehicle(GuardedMissionVehicle):
    """Inserted after RefinedMissionVehicle in the existing MRO.

    The frozen rescan algorithm remains unchanged; only its final executor changes.
    Recovery motor commands intentionally stay outside movement authorization.
    """
    def move(self,target_cell):
        if self.motor_latched_off:raise MotorInterlockError('motors are latched off')
        self.rescanning=True;before=self.state()
        try:
            result=rescan_and_move(_PostScanExecutor(self),self.sampling_history,self.confirmed_cell,target_cell,
                Intrinsics(40,30,25,25,19.5,14.5),assumption=self.sampling_assumption)
        finally:self.rescanning=False
        self.move_checks.append(result)
        if result['receipt'] is not None:
            if result['receipt']['completed']:self.confirmed_cell=tuple(target_cell)
            return result['receipt']
        receipt=rejected_receipt(self,before,(target_cell[0]+.5,target_cell[1]+.5,3.5),'RESCAN:'+result['state'])
        receipt.update(stop_confirmed=result['stop_confirmed'],sampling_checked=True)
        return receipt

    def move_after_scan(self,target_cell):
        before=self.state();target=(target_cell[0]+.5,target_cell[1]+.5,3.5)
        # Subclass runs actual model processing and returns a sealed small lease.
        lease,reason=self.observe_for_motion(target_cell)
        if lease is None:return rejected_receipt(self,before,target,reason)
        self._visual_lease=lease;self._motion_start=self.state()['time_s']
        self._motion_gate_active=True;self._commit_delay_used=False
        attempt=dict(target=list(target),start=self.state(),first_event=len(self.motor_checks))
        self.motion_attempts.append(attempt)
        try:
            result=PhysicalVehicle.move(self,target_cell)
        except MotionVeto as exc:
            self.hold_target=tuple(self.state()['position'])
            result=rejected_receipt(self,before,target,str(exc))
            self.actions.append(result)
        finally:
            self._motion_gate_active=False
            attempt.update(end=self.state(),last_event=len(self.motor_checks))
        attempt['receipt']=result
        return result

    def _step(self,target):
        pending=(tuple(target),self.state())
        previous=getattr(self,'_motor_pending',None)
        self._motor_pending=pending
        try:return super()._step(target)
        finally:self._motor_pending=previous

    def apply_motors(self,motors):
        if getattr(self,'_motion_gate_active',False):
            pending=getattr(self,'_motor_pending',None)
            if pending is None:raise MotionVeto('UNBOUND_MOTOR_COMMAND')
            target,command_state=pending
            if not self._commit_delay_used:
                self._commit_delay_used=True
                delay=self.spec['commit_delay_s']
                # Fault fixture between control calculation and actual submission.
                # Waiting applies the original hold controller, never stale motors.
                if delay:
                    self._motion_gate_active=False
                    try:self.hold(delay)
                    finally:self._motion_gate_active=True
            current=self.state()
            check=submission_check(self._visual_lease,target=target,now_s=current['time_s'],dt=self.physics.dt,
                clock_id='physics-seconds',command_state=command_state,current_state=current,
                move_started_at_s=self._motion_start,max_move_s=self.budget.max_move_s)
            self.motor_checks.append(dict(check,command_index=len(self.commands),before=current,
                command_state=command_state,target=list(target),proposed_motors=list(motors)))
            if not check['allowed']:raise MotionVeto(check['reason'])
        return super().apply_motors(motors)
