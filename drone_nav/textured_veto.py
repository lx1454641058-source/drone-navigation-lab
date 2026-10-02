"""来源：本项目原创。贴图相机的实际模型判断进入导航移动前否决。"""
from dataclasses import asdict,replace
from math import ceil,sqrt
from time import perf_counter

from .detection_bridge import number
from .pinhole import Intrinsics,Pose
from .packed_rgbd import PackedFrame
from .semantic_sensor import packet_for,context_for,visual_move_decision


def finish_visual_veto(candidate,packet,*,now_s,veto_enabled=True):
    """The experimental bypass disables only semantic target veto, never timing/context."""
    packet.validate()
    if (type(veto_enabled) is not bool or not number(now_s)
        or not number(packet.captured_at_s) or not number(packet.completed_at_s)
        or packet.clock_id!='physics-seconds' or now_s<packet.completed_at_s):
        raise ValueError('explicit comparison arm and valid completion time required')
    reason=candidate['reason']
    if reason not in ('VISUAL_CONTEXT_HOLD','VISUAL_TARGET_HOLD','NO_DETECTION_REQUIRES_GEOMETRY'):
        raise ValueError('unknown visual decision')
    if now_s>packet.captured_at_s+.5+1e-9:
        reason='VISUAL_EXPIRED_AT_COMMIT'
    elif not veto_enabled and reason=='VISUAL_TARGET_HOLD':
        reason='DEVELOPMENT_TARGET_VETO_DISABLED'
    return dict(reason=reason,permit_geometry_check=reason in ('NO_DETECTION_REQUIRES_GEOMETRY','DEVELOPMENT_TARGET_VETO_DISABLED'),
                checked_at_s=now_s,age_s=now_s-packet.captured_at_s,
                veto_enabled=veto_enabled,flight_authorized=False)


class PackedCameraDetector:
    """Convert the simulated immutable frame for the existing CPU session."""
    def __init__(self,session):
        self.session=session

    def detect(self,frame):
        started=perf_counter()
        packed=PackedFrame(frame.intrinsics,frame.pose,bytes(v for p in frame.rgb for v in p),tuple(frame.depth_z_m))
        original=self.session.detect_packed(packed)
        return dict(original,elapsed_s=perf_counter()-started,inner_elapsed_s=original['elapsed_s'])


class TexturedVetoMixin:
    def move(self,target_cell):
        if self.motor_latched_off:
            return super().move(target_cell)
        before=self.state()
        index=len(self.semantic_checks)
        frame_id=f'textured-semantic-{index}'
        raw_index=self.frame_index if self.saved is not None else len(self.frames)
        # Same sensor pose and resolution as the frozen semantic-move experiment.
        frame=self.capture(Intrinsics(160,120,100,100,79.5,59.5),0,
            lambda xyz:Pose.look_at(xyz,(target_cell[0]+.5,target_cell[1]+.5,.8)))
        paired=getattr(self,'paired_raw',None)
        if self.saved is not None:
            recorded=self.saved['semantic_checks'][index]
        elif paired is not None:
            recorded=paired['semantic_checks'][index]
            expected=paired['frames'][recorded['frame_index']]
            # This counterfactual may reuse a detection only for the exact capture.
            from json import dumps,loads
            if loads(dumps(asdict(frame)))!=expected['frame'] or before!=recorded['actual_before']:
                raise ValueError('paired input or capture state changed')
        else:
            recorded=None
        call=recorded['call'] if recorded is not None else self.detector.detect(frame)
        wait_s=call['elapsed_s']+self.spec['semantic_delay_s']
        wait_steps=ceil(wait_s/self.physics.dt)
        self.hold(wait_steps*self.physics.dt)
        available=self.state()['time_s']
        packet=packet_for(call,frame,frame_id=frame_id,captured_at_s=before['time_s'],completed_at_s=available)
        context=context_for(frame,frame_id,before['time_s'])
        if self.spec['semantic_wrong_frame']:
            context=replace(context,frame_id='wrong-textured-camera-frame')
        started=perf_counter()
        candidate=visual_move_decision(packet,context,now_s=available)
        measured=perf_counter()-started
        # Replays use the originally measured processing delay, never file-read speed.
        projection_s=recorded['projection_s'] if recorded is not None else measured
        projection_steps=ceil(projection_s/self.physics.dt)
        self.hold(projection_steps*self.physics.dt)
        committed=self.state()['time_s']
        decision=finish_visual_veto(candidate,packet,now_s=committed,veto_enabled=self.spec['semantic_veto_enabled'])
        check=dict(frame_index=raw_index,call=call,packet=asdict(packet),context=asdict(context),
            actual_before=before,actual_after=self.state(),waiting_steps=wait_steps,
            projection_s=projection_s,projection_waiting_steps=projection_steps,
            candidate=candidate,decision=decision,target=list(target_cell))
        self.semantic_checks.append(check)
        if decision['permit_geometry_check']:
            return super().move(target_cell)
        final=self.state()
        return dict(completed=False,reason=decision['reason'],start=before,end=final,
            target=(target_cell[0]+.5,target_cell[1]+.5,3.5),duration_s=final['time_s']-before['time_s'],
            speed_mps=sqrt(sum(v*v for v in final['velocity'])),stop_confirmed=False)
