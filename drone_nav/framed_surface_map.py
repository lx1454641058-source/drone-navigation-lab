"""来源：本项目原创。带参考坐标和时效的稀疏研究表面图，不推断空闲。"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from itertools import product
import json
from math import floor

from .frame_transform import RigidFrameTransform
from .detection_bridge import identifier,number
from .timed_inbox import TimedObservationInbox


def transform_fingerprint(value):
    payload=asdict(value) if type(value) is RigidFrameTransform else value
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class FramedSurfaceMap:
    """Sparse signed 3-D indices. Expired surfaces become unknown, never free.

    Map scope is fixed by the entire reference transform, not its target label.
    A capacity failure latches until the caller explicitly starts a new map.
    """
    def __init__(self,transform,*,camera_id,clock_id,resolution_m=.25,max_cells=4096):
        if type(transform) is not RigidFrameTransform or not all(identifier(v) for v in (camera_id,clock_id)):
            raise ValueError('validated transform and stream identities required')
        if not number(resolution_m) or not .01<=resolution_m<=10:
            raise ValueError('resolution outside 0.01..10 metres')
        if type(max_cells) is not int or not 1<=max_cells<=100000:
            raise ValueError('finite cell budget required')
        self.transform=transform; self.camera_id=camera_id; self.clock_id=clock_id
        self.resolution_m=resolution_m; self.max_cells=max_cells
        self.fingerprint=transform_fingerprint(transform)
        self._cells={}; self._latest=None; self._now=-1.; self._fault=None

    def _time(self,now,clock):
        if clock!=self.clock_id or not number(now) or now<0 or now<self._now:
            raise ValueError('map clock must match and remain monotonic')

    def cells_for_point(self,point):
        if not isinstance(point,(tuple,list)) or len(point)!=3 or not all(number(v) for v in point):
            raise ValueError('finite three dimensional point required')
        axes=[]
        for coordinate in point:
            value=coordinate/self.resolution_m
            if not number(value): raise ValueError('point exceeds index range')
            nearest=round(value)
            # Protect both sides of a grid plane, including negative/zero axes.
            axes.append((nearest-1,nearest) if abs(value-nearest)<=1e-8 else (floor(value),))
        return list(product(*axes))

    def _answer(self,reason,*,accepted=False,**extra):
        return dict(accepted=accepted,reason=reason,recorded_cells=len(self._cells),fault=self._fault,
                    navigation_map_update_allowed=False,flight_authorized=False,free_space_evidence=[],**extra)

    def ingest(self,finished,*,now_s,clock_id):
        self._time(now_s,clock_id)
        if type(finished) is not dict or type(finished.get('available')) is not bool:
            raise ValueError('finished conversion envelope required')
        if any(finished.get(k) is not False for k in ('navigation_map_update_allowed','physical_navigation_calibrated','flight_authorized')):
            raise ValueError('only non-authorizing research observations are accepted')
        if not number(finished.get('now_s')) or finished['now_s']<0:
            raise ValueError('conversion completion time required')
        if self._fault:
            self._now=now_s; return self._answer('MAP_FAULT_LATCHED')
        if not finished['available']:
            self._now=now_s; return self._answer('TRANSFORM_UNAVAILABLE')
        mapped=finished.get('result')
        if type(mapped) is not dict or mapped.get('converted') is not True:
            raise ValueError('converted observation required')
        for key in ('navigation_map_update_allowed','physical_navigation_calibrated','flight_authorized'):
            if mapped.get(key) is not False: raise ValueError('research authority differs')
        if mapped.get('free_space_evidence')!=[]: raise ValueError('free-space evidence not supported')
        reason=None
        if transform_fingerprint(mapped.get('transform'))!=self.fingerprint: reason='REFERENCE_TRANSFORM_MISMATCH'
        elif mapped.get('world_frame')!=self.transform.target_frame or finished.get('world_frame')!=self.transform.target_frame: reason='WORLD_FRAME_MISMATCH'
        elif mapped.get('source_world_frame')!=self.transform.source_frame: reason='SOURCE_FRAME_MISMATCH'
        elif mapped.get('clock_id')!=self.clock_id: reason='SOURCE_CLOCK_MISMATCH'
        elif finished['now_s']>now_s: reason='CONVERSION_NOT_READY'
        if reason:
            self._now=now_s; return self._answer(reason)
        source=mapped.get('observation')
        receiver=TimedObservationInbox(camera_id=self.camera_id,clock_id=self.clock_id,world_frame=self.transform.target_frame)
        check=receiver.submit(source,now_s=now_s,clock_id=clock_id)
        if not check['accepted']:
            self._now=now_s; return self._answer(check['reason'])
        p=source['projection']; capture=p['captured_at_s']
        if (mapped.get('frame_id')!=p['frame_id'] or mapped.get('captured_at_s')!=capture
                or mapped.get('valid_until_s')!=p['valid_until_s'] or mapped.get('checked_at_s')!=p['consumer_at_s']
                or not number(finished.get('age_s')) or abs(finished['age_s']-(finished['now_s']-capture))>1e-8
                or finished['now_s']<p['consumer_at_s']):
            raise ValueError('conversion identity/deadline differs')
        if self._latest and p['frame_id']==self._latest[0]: reason='REPEATED_FRAME'
        elif self._latest and capture<=self._latest[1]: reason='OUT_OF_ORDER_CAPTURE'
        if reason:
            self._now=now_s; return self._answer(reason)
        staged={}; samples=0
        for observation in p['observations']:
            d=observation.get('detection')
            if type(d) is not dict or type(d.get('id')) is not int or d['id']<0 or d.get('group') not in ('person','vehicle'):
                raise ValueError('sample detection provenance missing')
            for sample in observation['samples']:
                original=sample.get('source_world_m')
                if not isinstance(original,(tuple,list)) or len(original)!=3 or not all(number(v) for v in original):
                    raise ValueError('original world point required')
                expected=self.transform.point(original,source_frame=self.transform.source_frame,target_frame=self.transform.target_frame)
                if max(abs(a-b) for a,b in zip(expected,sample['world_m']))>1e-8:
                    raise ValueError('transformed point changed after validation')
                reference=dict(detection_id=d['id'],reported_group=d['group'],pixel=list(sample['pixel']),
                    world_m=list(sample['world_m']),source_world_m=list(original),association='unverified_box_surface')
                for cell in self.cells_for_point(sample['world_m']): staged.setdefault(cell,[]).append(reference)
                samples+=1
        if len(set(self._cells)|set(staged))>self.max_cells:
            self._fault='CELL_BUDGET_EXCEEDED'; self._now=now_s
            return self._answer('CELL_BUDGET_EXCEEDED')
        # Commit only after every point and the whole capacity requirement passed.
        for cell,refs in staged.items():
            self._cells[cell]=dict(frame_id=p['frame_id'],camera_id=self.camera_id,captured_at_s=capture,
                valid_until_s=p['valid_until_s'],references=deepcopy(refs),detector_source=p['detector_source'])
        self._latest=(p['frame_id'],capture); self._now=now_s
        return self._answer('SURFACE_EVIDENCE_RECORDED' if staged else 'NO_SURFACES_NO_CLEARING',accepted=True,
                            input_samples=samples,updated_cells=len(staged),cleared_cells=0)

    def snapshot(self,*,now_s,clock_id):
        self._time(now_s,clock_id); self._now=now_s
        cells=[dict(index=list(index),state='map_unavailable' if self._fault else 'current_surface_evidence' if now_s<=value['valid_until_s']+1e-9
                    else 'unknown_after_expiry',**deepcopy(value)) for index,value in sorted(self._cells.items())]
        return dict(world_frame=self.transform.target_frame,reference_fingerprint=self.fingerprint,clock_id=self.clock_id,
            now_s=now_s,resolution_m=self.resolution_m,recorded_cells=len(cells),max_cells=self.max_cells,fault=self._fault,
            unseen_state='unknown_unobserved',cells=cells,free_cells=0,flight_authorized=False,navigation_map_update_allowed=False)

    def state(self,index,*,now_s,clock_id):
        self._time(now_s,clock_id)
        if type(index) is not tuple or len(index)!=3 or any(type(v) is not int for v in index):
            raise ValueError('three integer cell indices required')
        self._now=now_s
        if self._fault: return 'map_unavailable'
        item=self._cells.get(index)
        return ('unknown_unobserved' if item is None else 'current_surface_evidence'
                if now_s<=item['valid_until_s']+1e-9 else 'unknown_after_expiry')
