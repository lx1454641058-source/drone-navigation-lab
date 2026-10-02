"""来源：本项目原创。主导航逐段移动前执行采样检查和实际悬停复查。"""
from dataclasses import replace
from math import sqrt

from .mission_supervisor import MissionVehicle
from .physical_rescan import PhysicalSamplingHistory, rescan_and_move
from .physical_vehicle import PhysicalVehicle
from .pinhole import Intrinsics
from .sampling_motion import SamplingAssumption, SamplingView


class _CheckedExecutor:
    """复用原复查器，只有其最终检查通过后才调用底层移动，避免递归。"""
    def __init__(self, vehicle):
        self.vehicle = vehicle

    def __getattr__(self, name):
        return getattr(self.vehicle, name)

    @property
    def hold_target(self):
        return self.vehicle.hold_target

    @hold_target.setter
    def hold_target(self, value):
        self.vehicle.hold_target = value

    def move(self, target):
        return PhysicalVehicle.move(self.vehicle, target)


class GuardedMissionVehicle(MissionVehicle):
    """沿用原任务主管；所有规划器 move 调用均经过独立的观测历史门槛。"""
    def __init__(self, *args, sampling_assumption=None, **kwargs):
        if sampling_assumption is not None and not isinstance(sampling_assumption, SamplingAssumption):
            raise ValueError('invalid sampling assumption')
        super().__init__(*args, **kwargs)
        start = args[1] if len(args) > 1 else kwargs['start']
        self.confirmed_cell = tuple(start)
        self.sampling_assumption = sampling_assumption
        self.sampling_history = PhysicalSamplingHistory(clock_id='physics-seconds')
        self.move_checks = []
        self.source_frames = []
        self._scan_records = None
        self.rescanning = False

    def capture(self, intrinsics, tick, aim=None):
        state = self.state()
        frame = super().capture(intrinsics, tick, aim)
        if self._scan_records is not None:
            self._scan_records.append((state, frame))
        return frame

    def scan(self, tick, aim):
        # 导航的八帧共享 batch tick；采样账本另有逐帧序号，不改原帧或实际时钟。
        self._scan_records = []
        try:
            frames = super().scan(tick, aim)
            available = self.state()['time_s']
            for state, frame in self._scan_records:
                ledger_tick = self.sampling_history.grid.last_tick + 1
                view = SamplingView(replace(frame, tick=ledger_tick), f'nav-{ledger_tick}',
                                    'camera', 'physics-seconds', state['time_s'], available)
                self.sampling_history.add(view, now_s=available)
                self.source_frames.append(dict(navigation_tick=frame.tick, ledger_tick=ledger_tick,
                                               captured_at_s=state['time_s'], available_at_s=available))
            return frames
        finally:
            self._scan_records = None

    def move(self, target_cell):
        if self.motor_latched_off:
            # Keep the same interlock behavior as all other motor paths.
            from .mission_supervisor import MotorInterlockError
            raise MotorInterlockError('motors are latched off')
        self.rescanning = True
        before = self.state()
        try:
            result = rescan_and_move(
                _CheckedExecutor(self), self.sampling_history, self.confirmed_cell, target_cell,
                Intrinsics(40, 30, 25, 25, 19.5, 14.5), assumption=self.sampling_assumption)
        finally:
            self.rescanning = False
        self.move_checks.append(result)
        if result['receipt'] is not None:
            if result['receipt']['completed']:
                self.confirmed_cell = tuple(target_cell)
            return result['receipt']
        final = self.state()
        return dict(completed=False, reason='RESCAN:'+result['state'], start=before, end=final,
                    target=(target_cell[0]+.5, target_cell[1]+.5, 3.5),
                    duration_s=final['time_s']-before['time_s'],
                    speed_mps=sqrt(sum(v*v for v in final['velocity'])),
                    stop_confirmed=result['stop_confirmed'], sampling_checked=True)
