"""来源：本项目原创。神经网络结果在原几何移动检查之前提供拒绝条件。"""
from dataclasses import asdict, replace
from math import ceil, sqrt
from time import perf_counter

from .pinhole import Intrinsics, Pose
from .semantic_sensor import packet_for, context_for, visual_move_decision


class SemanticMoveMixin:
    def move(self, target_cell):
        if self.motor_latched_off:
            return super().move(target_cell)
        before = self.state()
        index = len(self.semantic_checks)
        frame_id = f'semantic-{index}'
        raw_index = self.frame_index if self.saved is not None else len(self.frames)
        frame = self.capture(Intrinsics(160, 120, 100, 100, 79.5, 59.5), 0,
            lambda xyz: Pose.look_at(xyz, (target_cell[0]+.5, target_cell[1]+.5, .8)))
        started = perf_counter()
        if self.saved is None:
            try:
                record = self.detector.detect(frame)
            except (ValueError, OSError, RuntimeError) as exc:
                record = dict(elapsed_s=perf_counter()-started, error=str(exc))
        else:
            if index >= len(self.saved['semantic_checks']):
                raise ValueError('missing semantic observation')
            record = self.saved['semantic_checks'][index]['call']
        # Apply measured preprocessing, archive and inference latency to physics.
        # Saved replay uses the original duration, not the speed of reading a file.
        delay = record['elapsed_s'] + self.spec['semantic_delay_s']
        steps = ceil(delay / self.physics.dt)
        self.hold(steps * self.physics.dt)
        available = self.state()['time_s']
        packet = context = None
        if 'error' in record:
            decision = dict(permit_geometry_check=False, reason='VISUAL_INFERENCE_HOLD',
                            error=record['error'], flight_authorized=False)
        else:
            packet = packet_for(record, frame, frame_id=frame_id,
                captured_at_s=before['time_s'], completed_at_s=available)
            context = context_for(frame, frame_id, before['time_s'])
            if self.spec['semantic_wrong_frame']:
                context = replace(context, frame_id='mismatched-camera-frame')
            decision = visual_move_decision(packet, context, now_s=available)
        check = dict(frame_index=raw_index, call=record, packet=asdict(packet) if packet else None,
                     context=asdict(context) if context else None,
                     actual_before=before, actual_after=self.state(), waiting_steps=steps,
                     decision=decision, target=list(target_cell))
        self.semantic_checks.append(check)
        if decision['permit_geometry_check']:
            return super().move(target_cell)
        final = self.state()
        return dict(completed=False, reason=decision['reason'], start=before, end=final,
                    target=(target_cell[0]+.5,target_cell[1]+.5,3.5),
                    duration_s=final['time_s']-before['time_s'],
                    speed_mps=sqrt(sum(v*v for v in final['velocity'])), stop_confirmed=False)
