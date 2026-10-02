"""来源：本项目原创。实际物理悬停、随体相机复查与运动回执的内部联调。

策略只读取传感器、地图和物理状态，不读取世界几何。点采样条件仍不是空闲证明。
"""
from math import sqrt

from .detection_bridge import number
from .physical_vehicle import physical_guard, spatial_cells, rotate_vector
from .pinhole import Pose, PerspectiveFrame
from .sampling_motion import SamplingAssumption, SamplingView, inspect_voxel
from .stationary_rescan import SamplingHistory, next_phase, phase_pose


class PhysicalSamplingHistory(SamplingHistory):
    def check(self, start, end, *, now_s, budget, assumption=None):
        """保留物理控制器的完整预留区及停止期限，不使用旧运动学期限。"""
        self._time(now_s)
        if assumption is not None and not isinstance(assumption, SamplingAssumption):
            raise ValueError('invalid sampling assumption')
        tick = max(0, self.grid.last_tick)
        base = physical_guard(self.grid, start, end, tick, now_s, self.stamps, budget)
        radius = budget.body_radius_m + budget.tracking_margin_m + budget.stop_reserve_m
        cells = sorted(spatial_cells(start, end, radius))
        rows = []
        for cell in cells:
            checks = []
            for view, single in self._entries:
                reasons = []
                if single.state(cell, tick) != 'free':
                    reasons.append('NO_SINGLE_FRAME_FREE_EVIDENCE')
                if view.available_at_s > now_s:
                    reasons.append('FRAME_NOT_AVAILABLE')
                if base['latest_stop_s'] > view.captured_at_s + budget.free_ttl_s + 1e-9:
                    reasons.append('SAMPLING_EXPIRES_BEFORE_PHYSICAL_STOP')
                if assumption is None:
                    reasons.append('MINIMUM_FEATURE_ASSUMPTION_UNKNOWN')
                check = dict(frame_id=view.frame_id, camera_id=view.camera_id, reasons=reasons)
                if not reasons:
                    check = inspect_voxel(cell, view, assumption)
                check.update(captured_at_s=view.captured_at_s,
                             valid_until_s=view.captured_at_s + budget.free_ttl_s)
                checks.append(check)
            rows.append(dict(voxel=list(cell), supported=any(not c['reasons'] for c in checks),
                             candidates=checks))
        allowed = base['allowed'] and all(row['supported'] for row in rows)
        self._now = now_s
        return dict(allowed=allowed, baseline=base, required_cells=[list(c) for c in cells],
                    sampling=rows, reason=base['reason'] if not base['allowed'] else
                    'SAMPLING_CONDITION_HOLD' if not allowed else 'CONDITIONAL_PHYSICS_APPROVAL',
                    flight_authorized=False)


def state_error(state, target):
    return (sqrt(sum((a-b)**2 for a, b in zip(state['position'], target))),
            sqrt(sum(v*v for v in state['velocity'])))


def _hold_checked(vehicle, seconds, target, monitor):
    steps = round(seconds / vehicle.physics.dt)
    if steps < 1 or abs(steps * vehicle.physics.dt - seconds) > 1e-9:
        raise ValueError('duration must be a positive whole number of physical steps')
    for _ in range(steps):
        state = vehicle._step(vehicle.hold_target)
        error, speed = state_error(state, target)
        monitor['max_position_error_m'] = max(monitor['max_position_error_m'], error)
        monitor['max_speed_mps'] = max(monitor['max_speed_mps'], speed)
        if error > vehicle.budget.tracking_margin_m or speed > vehicle.budget.speed_tolerance_mps:
            return False
    return True


def rescan_and_move(vehicle, history, start, end, intrinsics, *, assumption=None,
                    mode='phase_diverse', max_scans=4, settle_s=.1, processing_s=.05):
    """实际采集时刻取自物理引擎；确认到达之前不推进离散位置。"""
    if type(max_scans) is not int or not 1 <= max_scans <= 4:
        raise ValueError('invalid scan count')
    if mode not in ('repeat', 'phase_diverse'):
        raise ValueError('invalid phase policy')
    for duration in (settle_s, processing_s):
        if not number(duration) or duration <= 0:
            raise ValueError('invalid scan duration')
        steps = round(duration / vehicle.physics.dt)
        if steps < 1 or abs(steps * vehicle.physics.dt-duration) > 1e-9:
            raise ValueError('duration must fit physics steps')
    # 同时检查相邻格协议；相机轴在机体参考系中由相位决定。
    phase_pose(start, end, intrinsics, (0., 0.))
    target = (start[0]+.5, start[1]+.5, 3.5)
    before = vehicle.state()
    error, speed = state_error(before, target)
    monitor = dict(max_position_error_m=error, max_speed_mps=speed)
    events, used = [], []
    decision = initial = None
    receipt = None
    state = 'INITIAL_POSE_HOLD'
    if error <= vehicle.budget.position_tolerance_m and speed <= vehicle.budget.speed_tolerance_mps:
        initial = decision = history.check(start, end, now_s=before['time_s'],
                                           budget=vehicle.budget, assumption=assumption)
        state = 'KNOWN_OBSTACLE_HOLD' if any(tuple(c) in history.grid.occupied
                                           for c in decision['required_cells']) else 'SCANNING'
    if state == 'SCANNING':
        for _ in range(max_scans):
            phase = next_phase(used, mode)
            used.append(phase)
            if not _hold_checked(vehicle, settle_s, target, monitor):
                state = 'SCAN_STABILITY_HOLD'
                break
            capture_state = vehicle.state()
            base = phase_pose(start, end, intrinsics, phase)
            def aim(xyz):
                return Pose(xyz, base.right, base.down, base.forward)
            tick = history.grid.last_tick + 1
            try:
                frame = vehicle.capture(intrinsics, tick, aim)
                if not isinstance(frame, PerspectiveFrame):
                    raise ValueError('camera did not return a frame')
                q = capture_state['quaternion']
                expected = Pose(tuple(capture_state['position']), rotate_vector(base.right, q),
                                rotate_vector(base.down, q), rotate_vector(base.forward, q))
                if frame.pose != expected or frame.tick != tick or frame.intrinsics != intrinsics:
                    raise ValueError('camera pose, tick or calibration differs from capture state')
                frame.validate()
                if not _hold_checked(vehicle, processing_s, target, monitor):
                    state = 'SCAN_STABILITY_HOLD'
                    events.append(dict(phase=list(phase), capture_state=capture_state, state=state))
                    break
                available = vehicle.state()['time_s']
                view = SamplingView(frame, 'physical-'+str(tick), 'camera', history.clock_id,
                                    capture_state['time_s'], available)
                info = history.add(view, now_s=available)
                decision = history.check(start, end, now_s=available, budget=vehicle.budget,
                                         assumption=assumption)
                positive = any(tuple(c) in history.grid.occupied for c in decision['required_cells'])
                state = 'OBSERVED_OBSTACLE_HOLD' if positive else (
                    'SCAN_SENSOR_HOLD' if info['valid_depth_fraction'] < .05 else 'SCANNING')
                events.append(dict(phase=list(phase), capture_state=capture_state,
                                   available_at_s=available, state=state, observation=info, decision=decision))
            except (ValueError, TypeError, OverflowError) as exc:
                state = 'SCAN_PROTOCOL_HOLD'
                events.append(dict(phase=list(phase), capture_state=capture_state,
                                   state=state, error=str(exc)))
            if state != 'SCANNING':
                break
        else:
            state = 'READY_FOR_PHYSICAL_MOVE' if decision['allowed'] else 'SCAN_BUDGET_HOLD'
    scan_end = vehicle.state()
    if state == 'READY_FOR_PHYSICAL_MOVE':
        error, speed = state_error(scan_end, target)
        if error > vehicle.budget.position_tolerance_m or speed > vehicle.budget.speed_tolerance_mps:
            state = 'PREMOVE_POSE_HOLD'
        else:
            receipt = vehicle.move(end)
            state = 'ARRIVED_AND_SLOW' if receipt['completed'] else receipt['reason']
    if receipt is None:
        # 拒绝新航点后以当前位置为悬停目标，并报告实际速度；拒绝不等于已停止。
        vehicle.hold_target = tuple(vehicle.state()['position'])
        vehicle.hold(.5)
    final = vehicle.state()
    return dict(state=state, initial=initial, final_guard=decision, events=events,
                receipt=receipt, actual_start=before, actual_after_scan=scan_end, actual_final=final,
                scan_duration_s=scan_end['time_s']-before['time_s'], monitor=monitor,
                stop_confirmed=state_error(final, target)[1] <= vehicle.budget.speed_tolerance_mps,
                confirmed_cell=list(end if receipt and receipt['completed'] else start),
                flight_authorized=False,
                limitation='实际 MuJoCo 状态、理想转台和中心 RGB-D；精确定位、静态场景和尺寸假设，非实机保证。')
