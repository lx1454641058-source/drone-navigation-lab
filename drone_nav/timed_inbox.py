"""来源：本项目原创。有界视觉观测接收器；不变换坐标、不授权飞行。"""
from collections import deque
from copy import deepcopy

from .detection_bridge import identifier,number


class TimedObservationInbox:
    """One camera/clock/world stream, with a bounded FIFO and a latched overflow.

    Rejecting or consuming an observation never clears previously mapped evidence.
    The receiver owns snapshots, so producer/caller mutation cannot refresh data.
    """
    def __init__(self,*,camera_id,clock_id,world_frame,capacity=2):
        if not all(identifier(v) for v in (camera_id,clock_id,world_frame)):
            raise ValueError('explicit camera, clock and world identities required')
        if type(capacity) is not int or not 1<=capacity<=64:
            raise ValueError('capacity must be an integer in 1..64')
        self.camera_id=camera_id; self.clock_id=clock_id; self.world_frame=world_frame
        self.capacity=capacity; self._pending=deque(); self._now=-1.
        self._latest=None; self._fault=None

    def _time(self,now,clock):
        if clock!=self.clock_id or not number(now) or now<0 or now<self._now:
            raise ValueError('consumer clock must match and time must be monotonic')

    def _answer(self,reason,**extra):
        return dict(reason=reason,pending=len(self._pending),fault=self._fault,
                    flight_authorized=False,navigation_map_update_allowed=False,
                    clears_previous_evidence=False,free_space_evidence=[],**extra)

    def _validate(self,result):
        if (type(result) is not dict or result.get('flight_authorized') is not False
                or result.get('navigation_frame_alignment_available') is not False):
            raise ValueError('unaligned non-authorizing projection required')
        p=result.get('projection')
        if type(p) is not dict or not all(identifier(p.get(k)) for k in ('frame_id','camera_id','clock_id','consumer_clock_id','detector_source')):
            raise ValueError('projection identity missing')
        keys=('captured_at_s','completed_at_s','consumer_at_s','valid_until_s','age_s')
        if any(not number(p.get(k)) or p[k]<0 for k in keys):
            raise ValueError('finite projection times required')
        if not p['captured_at_s']<=p['completed_at_s']<=p['consumer_at_s']:
            raise ValueError('projection time order differs')
        if abs(p['valid_until_s']-p['captured_at_s']-.5)>1e-8 or abs(p['age_s']-(p['consumer_at_s']-p['captured_at_s']))>1e-8:
            raise ValueError('capture age or fixed half-second deadline differs')
        if p.get('flight_authorized') is not False or p.get('free_space_evidence')!=[]:
            raise ValueError('projection cannot authorize or claim free space')
        if type(p.get('observations')) is not list or len(p['observations'])>30000:
            raise ValueError('invalid observations')
        count=0
        for o in p['observations']:
            if type(o) is not dict or type(o.get('samples')) is not list or len(o['samples'])>256:
                raise ValueError('invalid surface sample list')
            if o.get('association')!='unverified_box_surface' or o.get('object_extent_known') is not False:
                raise ValueError('sample association cannot certify object extent')
            for sample in o['samples']:
                if type(sample) is not dict: raise ValueError('invalid sample')
                point=sample.get('world_m'); pixel=sample.get('pixel'); depth=sample.get('depth_z_m')
                if (not isinstance(point,(tuple,list)) or len(point)!=3 or not all(number(x) for x in point)
                        or not isinstance(pixel,(tuple,list)) or len(pixel)!=2
                        or any(type(x) is not int or x<0 for x in pixel) or not number(depth) or depth<=0):
                    raise ValueError('invalid sample geometry')
                count+=1
        status=p.get('status')
        if status not in ('CONTEXT_REJECTED','SAMPLES_AVAILABLE','NO_SPATIAL_SAMPLES'):
            raise ValueError('unknown projection status')
        if status=='CONTEXT_REJECTED':
            if count or not p.get('reasons') or result.get('reason')!='VISUAL_CONTEXT_HOLD':
                raise ValueError('rejected projection contains usable evidence')
        else:
            if not identifier(p.get('world_frame')):
                raise ValueError('usable samples require an explicit source world')
            if p.get('reasons')!=[] or (status=='SAMPLES_AVAILABLE')!=(count>0):
                raise ValueError('projection sample status differs')
            expected='VISUAL_TARGET_HOLD' if p['observations'] else 'NO_DETECTION_REQUIRES_GEOMETRY'
            if result.get('reason')!=expected: raise ValueError('detection decision differs')
        return p

    def submit(self,result,*,now_s,clock_id):
        self._time(now_s,clock_id)
        p=self._validate(result)
        # Validation and copying happen before commit: malformed payloads leave
        # both the clock watermark and the pending queue unchanged.
        reason=None
        if self._fault: reason='STREAM_FAULT_LATCHED'
        elif p['camera_id']!=self.camera_id: reason='CAMERA_MISMATCH'
        elif p['clock_id']!=self.clock_id or p['consumer_clock_id']!=self.clock_id: reason='SOURCE_CLOCK_MISMATCH'
        elif p['consumer_at_s']>now_s: reason='RESULT_NOT_READY'
        elif p['status']=='CONTEXT_REJECTED': reason='CONTEXT_REJECTED'
        elif p['world_frame']!=self.world_frame: reason='SOURCE_WORLD_MISMATCH'
        elif self._latest and p['frame_id']==self._latest[0]: reason='REPEATED_FRAME'
        elif self._latest and p['captured_at_s']<=self._latest[1]: reason='OUT_OF_ORDER_CAPTURE'
        elif now_s>p['captured_at_s']+.5+1e-9: reason='EXPIRED_AT_RECEIPT'
        elif len(self._pending)>=self.capacity: reason='QUEUE_OVERFLOW'
        if reason:
            self._now=now_s
            if reason=='QUEUE_OVERFLOW': self._fault=reason
            return self._answer(reason,accepted=False,frame_id=p['frame_id'])
        owned=deepcopy(result)
        self._pending.append(owned); self._latest=(p['frame_id'],p['captured_at_s']); self._now=now_s
        return self._answer('QUEUED',accepted=True,frame_id=p['frame_id'])

    def consume(self,*,now_s,clock_id,target_world):
        self._time(now_s,clock_id)
        if not identifier(target_world): raise ValueError('explicit target world required')
        self._now=now_s
        if self._fault: return self._answer('STREAM_FAULT_LATCHED',delivered=False)
        # A caller aimed at another map must not steal or relabel queued data.
        if target_world!=self.world_frame:
            return self._answer('TARGET_WORLD_MISMATCH',delivered=False)
        if not self._pending: return self._answer('NO_PENDING_OBSERVATION',delivered=False)
        result=self._pending.popleft(); p=result['projection']
        if now_s>p['captured_at_s']+.5+1e-9:
            return self._answer('EXPIRED_AT_CONSUMPTION',delivered=False,frame_id=p['frame_id'])
        return self._answer(result['reason'],delivered=True,frame_id=p['frame_id'],
                            consumed_at_s=now_s,world_frame=self.world_frame,observation=deepcopy(result))

    def status(self):
        return dict(pending=len(self._pending),capacity=self.capacity,fault=self._fault,now_s=self._now,
                    latest=None if self._latest is None else list(self._latest),flight_authorized=False)
