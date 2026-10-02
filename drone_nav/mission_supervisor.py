"""来源：本项目原创。统一合成视觉导航、下降与异常恢复，停桨后禁止重新驱动。"""
from dataclasses import asdict
from math import sqrt

from .descent_vehicle import DescentVehicle
from .detection_bridge import number
from .hover_recovery import RecoveryBudget, recover_hover
from .physical_navigation import navigate_physical
from .pinhole import finite_vector
from .visual_descent import DescentConfig, descend_after_navigation


class MotorInterlockError(RuntimeError):
    pass


class MissionVehicle(DescentVehicle):
    """锁针对本实例的所有控制路径；不得以“恢复”为理由在停桨后重新启动。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motor_latched_off = False
        self.disarm_requested_at_s = None
        self.operation_phase = 'CREATED'

    def apply_motors(self, motors):
        if (not isinstance(motors, (list, tuple)) or len(motors) != 4
                or any(not number(v) or not 0 <= v <= 8 for v in motors)):
            raise ValueError('four finite motor commands within 0..8 required')
        if self.motor_latched_off and any(v != 0 for v in motors):
            raise MotorInterlockError('motors are latched off; rearming requires a new supervised procedure')
        if not self.motor_latched_off and all(v == 0 for v in motors):
            self.disarm_requested_at_s = self.state()['time_s']
            self.motor_latched_off = True
        return super().apply_motors(motors)


TRANSITIONS = {
    'CREATED': {'CRUISING', 'DISARMED_UNCONFIRMED', 'MANUAL_REVIEW'},
    'CRUISING': {'LANDING_AUTHORIZED', 'ABORTING', 'DISARMED_UNCONFIRMED', 'MANUAL_REVIEW'},
    'LANDING_AUTHORIZED': {'DESCENDING', 'ABORTING', 'DISARMED_UNCONFIRMED', 'MANUAL_REVIEW'},
    'DESCENDING': {'LANDED_CONFIRMED', 'DISARMED_UNCONFIRMED', 'ABORTING', 'MANUAL_REVIEW'},
    'ABORTING': {'RECOVERING', 'DISARMED_UNCONFIRMED', 'MANUAL_REVIEW'},
    'RECOVERING': {'ABORTED_MOTION_STABLE', 'ABORTED_RECOVERY_UNCONFIRMED',
                   'DISARMED_UNCONFIRMED', 'MANUAL_REVIEW'},
}


class MissionSupervisor:
    """一个实例只运行一次任务；终态不能隐式重新进入巡航。"""
    def __init__(self, vehicle, model, start, goal, *, max_ticks=40,
                 descent_config=None, recovery_budget=None):
        for cell in (start, goal):
            if (not isinstance(cell, (tuple, list)) or len(cell) != 2
                    or any(type(v) is not int for v in cell)
                    or not 1 <= cell[0] < 19 or not 1 <= cell[1] < 15):
                raise ValueError('interior start and goal cells required')
        if type(max_ticks) is not int or not 1 <= max_ticks <= 100:
            raise ValueError('invalid navigation budget')
        if type(getattr(vehicle, 'motor_latched_off', None)) is not bool:
            raise ValueError('supervised vehicle with motor interlock required')
        self.descent_config = DescentConfig() if descent_config is None else descent_config
        self.recovery_budget = RecoveryBudget() if recovery_budget is None else recovery_budget
        if not isinstance(self.descent_config, DescentConfig) or not isinstance(self.recovery_budget, RecoveryBudget):
            raise ValueError('invalid mission configuration')
        self.vehicle, self.model = vehicle, model
        self.start, self.goal = tuple(start), tuple(goal)
        self.max_ticks, self.state = max_ticks, 'CREATED'
        self.events = []
        self.navigation = self.descent = self.recovery = None
        self.confirmed_cell = self.start
        self.original_reason = None
        self._started = False

    def snapshot(self):
        try:
            state = self.vehicle.state()
            if (not isinstance(state, dict) or not number(state.get('time_s')) or state['time_s'] < 0
                    or any(not finite_vector(state.get(k), n) for k,n in
                           (('position',3),('velocity',3),('quaternion',4),('angular_velocity',3)))
                    or abs(sum(v*v for v in state['quaternion'])-1) > 1e-6):
                return None
            return state
        except (ValueError, TypeError, RuntimeError, OverflowError):
            return None

    def transition(self, state, reason):
        if state not in TRANSITIONS.get(self.state, set()):
            raise RuntimeError('invalid mission transition: '+self.state+' -> '+state)
        old = self.state
        self.state = state
        self.vehicle.operation_phase = state
        actual = self.snapshot()
        self.events.append(dict(previous=old, state=state, reason=reason, actual=actual,
                                time_s=None if actual is None else actual['time_s'],
                                motors_off=self.vehicle.motor_latched_off))

    def abort(self, reason):
        self.original_reason = reason
        if self.vehicle.motor_latched_off:
            self.transition('DISARMED_UNCONFIRMED', reason)
            return
        self.transition('ABORTING', reason)
        if self.snapshot() is None:
            self.transition('MANUAL_REVIEW', '当前物理状态不可读，不能确认恢复')
            return
        self.transition('RECOVERING', '保持原悬停目标，继续验证连续运动状态')
        try:
            self.recovery = recover_hover(self.vehicle, budget=self.recovery_budget)
            state = ('DISARMED_UNCONFIRMED' if self.vehicle.motor_latched_off else
                     'ABORTED_MOTION_STABLE' if self.recovery['motion_stable'] else 'ABORTED_RECOVERY_UNCONFIRMED')
            self.transition(state, self.recovery['state'])
        except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as exc:
            # 若控制期间已经停桨，绝不能再尝试一次带推力的恢复。
            state = 'DISARMED_UNCONFIRMED' if self.vehicle.motor_latched_off else 'ABORTED_RECOVERY_UNCONFIRMED'
            self.transition(state, str(exc))

    def run(self):
        if self._started:
            raise RuntimeError('mission is single-use; terminal state cannot be resumed implicitly')
        self._started = True
        if self.vehicle.motor_latched_off:
            self.transition('DISARMED_UNCONFIRMED', '任务启动前已停桨，不重新启动')
            return self.result()
        initial = self.snapshot()
        if initial is None:
            self.transition('MANUAL_REVIEW', '初始物理状态不可读')
            return self.result()
        target = (self.start[0]+.5, self.start[1]+.5, 3.5)
        if (sqrt(sum((a-b)**2 for a,b in zip(initial['position'], target))) > .03
                or sqrt(sum(v*v for v in initial['velocity'])) > .03):
            self.transition('MANUAL_REVIEW', '初始位置或速度不满足任务前提，不发出航点')
            return self.result()
        self.transition('CRUISING', '开始原合成视觉导航与物理到达确认')
        try:
            self.navigation = navigate_physical(self.vehicle, self.model, self.start, self.goal,
                                                max_ticks=self.max_ticks, previews=False)
            if self.navigation['trace']:
                self.confirmed_cell = tuple(self.navigation['trace'][-1]['position'])
            if self.navigation['terminal_state'] != 'READY_TO_LAND':
                self.abort('NAVIGATION:'+self.navigation['terminal_state'])
                return self.result()
            self.transition('LANDING_AUTHORIZED', '到达与末端视觉通过，继续复查下降通道')
            self.transition('DESCENDING', '开始原视觉下降控制')
            self.descent = descend_after_navigation(self.vehicle, self.model, self.navigation,
                                                    config=self.descent_config, previews=False)
            status = self.descent['status']
            if status == 'LANDED' and self.descent['stop_confirmed'] and self.vehicle.motor_latched_off:
                self.transition('LANDED_CONFIRMED', '停桨后接地持续确认；尚未放餐或返航')
            elif self.vehicle.motor_latched_off or self.descent['disarmed_at_s'] is not None:
                self.original_reason = 'DESCENT:'+status
                self.transition('DISARMED_UNCONFIRMED', '已停桨但接地未确认，保持关闭并等待检查')
            else:
                self.abort('DESCENT:'+status+':'+self.descent['reason'])
        except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as exc:
            if self.state in ('CRUISING', 'LANDING_AUTHORIZED', 'DESCENDING'):
                self.abort('PROTOCOL:'+str(exc))
            else:
                raise
        return self.result()

    def result(self):
        return dict(state=self.state, original_reason=self.original_reason, start=list(self.start),
                    goal=list(self.goal), confirmed_cell=list(self.confirmed_cell), events=self.events,
                    navigation=self.navigation, descent=self.descent, recovery=self.recovery,
                    actual_final=self.snapshot(), motors_off=self.vehicle.motor_latched_off,
                    disarm_requested_at_s=self.vehicle.disarm_requested_at_s,
                    touchdown_confirmed=self.state=='LANDED_CONFIRMED', delivery_completed=False,
                    resume_allowed=False, flight_authorized=False,
                    descent_config=asdict(self.descent_config), recovery_budget=asdict(self.recovery_budget),
                    limitation='统一原合成视觉流程及异常恢复；不含真实识别控制、起飞、放餐、返航或实机授权。')
