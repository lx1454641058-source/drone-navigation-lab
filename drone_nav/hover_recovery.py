"""来源：本项目原创。任务中止后基于连续物理反馈确认悬停，不重新授权飞行。"""
from dataclasses import asdict, dataclass
from math import acos, degrees, sqrt

from .detection_bridge import number
from .pinhole import finite_vector
from .physical_rescan import rescan_and_move


@dataclass(frozen=True)
class RecoveryBudget:
    max_duration_s: float = 4.
    stable_s: float = .3
    speed_mps: float = .03
    position_error_m: float = .03
    angular_speed_rad_s: float = .05
    tilt_deg: float = 5.
    max_drift_m: float = .12

    def __post_init__(self):
        if any(not number(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('recovery budgets must be finite and positive')
        if (self.stable_s > self.max_duration_s or self.max_duration_s > 10
                or self.position_error_m >= self.max_drift_m or self.tilt_deg >= 90):
            raise ValueError('inconsistent recovery budget')


def measure(state, target, origin, budget):
    if not isinstance(state, dict) or not number(state.get('time_s')) or state['time_s'] < 0:
        raise ValueError('invalid recovery timestamp')
    for key, size in (('position', 3), ('velocity', 3), ('quaternion', 4), ('angular_velocity', 3)):
        if not finite_vector(state.get(key), size):
            raise ValueError('invalid recovery '+key)
    q = state['quaternion']
    if abs(sum(v*v for v in q)-1) > 1e-6:
        raise ValueError('invalid quaternion norm')
    norm = lambda v: sqrt(sum(x*x for x in v))
    speed = norm(state['velocity'])
    angular = norm(state['angular_velocity'])
    error = norm([a-b for a, b in zip(state['position'], target)])
    drift = norm([a-b for a, b in zip(state['position'], origin)])
    tilt = degrees(acos(max(-1., min(1., 1-2*(q[1]*q[1]+q[2]*q[2])))))
    return dict(time_s=state['time_s'], speed_mps=speed, angular_speed_rad_s=angular,
                position_error_m=error, drift_m=drift, tilt_deg=tilt,
                qualified=(speed <= budget.speed_mps and angular <= budget.angular_speed_rad_s
                           and error <= budget.position_error_m and tilt <= budget.tilt_deg),
                envelope_ok=max(error, drift) <= budget.max_drift_m)


def recover_hover(vehicle, *, budget=None):
    """固定已有悬停目标；连续合格稳定窗口通过后才确认，坏状态不伪造停稳。"""
    budget = RecoveryBudget() if budget is None else budget
    if not isinstance(budget, RecoveryBudget):
        raise ValueError('invalid recovery budget')
    dt = vehicle.physics.dt
    if not number(dt) or dt <= 0:
        raise ValueError('invalid physics step')
    for duration in (budget.max_duration_s, budget.stable_s):
        if abs(round(duration/dt)*dt-duration) > 1e-9:
            raise ValueError('recovery time budget must fit physical steps')
    target = tuple(vehicle.hold_target)
    if not finite_vector(target, 3):
        raise ValueError('invalid fixed hover target')
    states, samples = [], []
    stable_since = None
    stable_duration = 0.
    current_verified = False
    terminal, error = 'RECOVERY_STATE_UNAVAILABLE', None
    try:
        first = vehicle.state()
        origin = tuple(first['position'])
        m = measure(first, target, origin, budget)
        states.append(first); samples.append(m)
        current_verified = True
        start = first['time_s']
        if not m['envelope_ok']:
            terminal = 'RECOVERY_ENVELOPE_BREACH'
        else:
            stable_since = start if m['qualified'] else None
            terminal = 'RECOVERY_TIMEOUT'
            for _ in range(round(budget.max_duration_s/dt)):
                current_verified = False
                state = vehicle._step(target)
                m = measure(state, target, origin, budget)
                if abs(state['time_s'] - states[-1]['time_s'] - dt) > 1e-8:
                    terminal = 'RECOVERY_CLOCK_FAULT'
                    break
                states.append(state); samples.append(m)
                current_verified = True
                if not m['envelope_ok']:
                    terminal = 'RECOVERY_ENVELOPE_BREACH'
                    break
                if m['qualified']:
                    stable_since = m['time_s'] if stable_since is None else stable_since
                else:
                    stable_since = None
                stable_duration = 0. if stable_since is None else m['time_s']-stable_since
                if stable_duration + 1e-9 >= budget.stable_s:
                    terminal = 'MOTION_STABLE'
                    break
    except (ValueError, TypeError, KeyError, OverflowError, RuntimeError) as exc:
        terminal, error = 'RECOVERY_STATE_UNAVAILABLE', str(exc)
    return dict(state=terminal, motion_stable=terminal=='MOTION_STABLE', target=list(target),
                stable_duration_s=stable_duration, budget=asdict(budget), error=error,
                current_state_verified=current_verified,
                last_verified_state=states[-1] if states else None,
                elapsed_verified_s=states[-1]['time_s']-states[0]['time_s'] if states else 0.,
                states=states, samples=samples, resume_allowed=False, flight_authorized=False,
                spatial_safety_certified=False,
                limitation='仅确认有限采样间隔内的局部悬停状态；不清除障碍、历史失败或重新批准路线。')


def finish_aborted_operation(vehicle, operation, *, budget=None):
    """把恢复结果附加到原失败，不把已停稳写成配送或导航成功。"""
    if not isinstance(operation, dict) or operation.get('actual_final') != vehicle.state():
        raise ValueError('operation receipt does not match current vehicle state')
    receipt = operation.get('receipt')
    if operation.get('state') == 'ARRIVED_AND_SLOW' and receipt and receipt.get('completed'):
        return dict(state='ARRIVED_AND_SLOW', operation=operation, recovery=None,
                    resume_allowed=False, flight_authorized=False)
    recovery = recover_hover(vehicle, budget=budget)
    return dict(state='ABORTED_MOTION_STABLE' if recovery['motion_stable'] else
                'ABORTED_RECOVERY_UNCONFIRMED', original_state=operation['state'],
                confirmed_cell=operation['confirmed_cell'], operation=operation, recovery=recovery,
                resume_allowed=False, flight_authorized=False)


def rescan_with_recovery(vehicle, history, start, end, intrinsics, *, recovery_budget=None, **kwargs):
    operation = rescan_and_move(vehicle, history, start, end, intrinsics, **kwargs)
    return finish_aborted_operation(vehicle, operation, budget=recovery_budget)
