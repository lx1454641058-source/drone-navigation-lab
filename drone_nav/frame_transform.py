"""来源：本项目原创。显式米制刚体变换与研究坐标观测，不授权导航。"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import re

from .detection_bridge import identifier,number
from .pinhole import Pose,Intrinsics,dot,cross
from .timed_inbox import TimedObservationInbox


def vector(value):
    return type(value) is tuple and len(value)==3 and all(number(v) for v in value)


@dataclass(frozen=True)
class RigidFrameTransform:
    source_frame: str
    target_frame: str
    rotation: tuple
    translation_m: tuple
    reference_id: str
    reference_sha256: str
    unit: str='m'

    def __post_init__(self):
        if (not all(identifier(v) for v in (self.source_frame,self.target_frame,self.reference_id))
                or self.source_frame==self.target_frame or self.unit!='m'
                or type(self.reference_sha256) is not str or re.fullmatch('[0-9a-f]{64}',self.reference_sha256) is None):
            raise ValueError('distinct frames, metre units and reference digest required')
        if (type(self.rotation) is not tuple or len(self.rotation)!=3
                or not all(vector(row) for row in self.rotation) or not vector(self.translation_m)):
            raise ValueError('immutable finite 3x3 rotation and translation required')
        r=self.rotation
        if any(abs(dot(r[i],r[j])-(1 if i==j else 0))>1e-8 for i in range(3) for j in range(3)) or abs(dot(cross(r[0],r[1]),r[2])-1)>1e-8:
            raise ValueError('rotation must be orthonormal and right handed; scale/reflection rejected')

    def _frames(self,source_frame,target_frame,unit):
        if (source_frame,target_frame,unit)!=(self.source_frame,self.target_frame,'m'):
            raise ValueError('transform direction, frame or unit mismatch')

    def direction(self,value,*,source_frame,target_frame,unit='m'):
        self._frames(source_frame,target_frame,unit)
        value=tuple(value)
        if not vector(value): raise ValueError('finite vector required')
        return tuple(dot(row,value) for row in self.rotation)

    def point(self,value,*,source_frame,target_frame,unit='m'):
        rotated=self.direction(value,source_frame=source_frame,target_frame=target_frame,unit=unit)
        return tuple(a+b for a,b in zip(rotated,self.translation_m))

    def pose(self,pose,*,source_frame,target_frame,unit='m'):
        if type(pose) is not Pose: raise ValueError('validated camera pose required')
        options=dict(source_frame=source_frame,target_frame=target_frame,unit=unit)
        return Pose(self.point(pose.position,**options),*(self.direction(a,**options) for a in (pose.right,pose.down,pose.forward)))

    def inverse(self):
        r=tuple(tuple(self.rotation[j][i] for j in range(3)) for i in range(3))
        return RigidFrameTransform(self.target_frame,self.source_frame,r,
            tuple(-dot(row,self.translation_m) for row in r),self.reference_id+':inverse',self.reference_sha256)

    @classmethod
    def from_reference_pose(cls,pose,*,source_frame,target_frame,reference_id,reference_sha256):
        if type(pose) is not Pose: raise ValueError('reference camera pose required')
        # Camera-to-world orientation is stored as columns; the inverse has
        # camera axes as rows. The anchor maps to (0,0,0), without assuming gravity.
        r=tuple(tuple(a) for a in (pose.right,pose.down,pose.forward))
        return cls(source_frame,target_frame,r,tuple(-dot(a,pose.position) for a in r),reference_id,reference_sha256)


@dataclass(frozen=True)
class CameraPoseStamp:
    frame_id: str
    camera_id: str
    clock_id: str
    pose_at_s: float
    pose: Pose
    intrinsics: Intrinsics

    def __post_init__(self):
        if (not all(identifier(v) for v in (self.frame_id,self.camera_id,self.clock_id))
                or not number(self.pose_at_s) or self.pose_at_s<0
                or type(self.pose) is not Pose or type(self.intrinsics) is not Intrinsics):
            raise ValueError('camera geometry identity and time required')
        if not all(vector(v) for v in (self.pose.position,self.pose.right,self.pose.down,self.pose.forward)):
            raise ValueError('immutable camera pose required')


def transform_delivery(delivery,transform,geometry,*,now_s,clock_id,target_frame):
    if type(transform) is not RigidFrameTransform or type(geometry) is not CameraPoseStamp:
        raise ValueError('explicit validated transform and camera geometry required')
    if (type(delivery) is not dict or delivery.get('delivered') is not True
            or delivery.get('flight_authorized') is not False or delivery.get('navigation_map_update_allowed') is not False
            or delivery.get('clears_previous_evidence') is not False or delivery.get('free_space_evidence')!=[]):
        raise ValueError('non-authorizing delivered observation required')
    transform._frames(delivery.get('world_frame'),target_frame,'m')
    consumed=delivery.get('consumed_at_s')
    if not number(now_s) or not number(consumed) or now_s<consumed:
        raise ValueError('conversion cannot precede consumption')
    source=delivery.get('observation'); receiver=TimedObservationInbox(camera_id=geometry.camera_id,
        clock_id=geometry.clock_id,world_frame=transform.source_frame)
    check=receiver.submit(source,now_s=now_s,clock_id=clock_id)
    if not check['accepted']: raise ValueError('conversion input rejected: '+check['reason'])
    p=source['projection']
    if (delivery.get('frame_id')!=p['frame_id'] or geometry.frame_id!=p['frame_id']
            or delivery.get('reason')!=source['reason'] or consumed<p['consumer_at_s']
            or abs(geometry.pose_at_s-p['captured_at_s'])>.02+1e-9 or geometry.pose_at_s>now_s):
        raise ValueError('camera frame, pose time or delivered identity differs')
    options=dict(source_frame=transform.source_frame,target_frame=transform.target_frame)
    out=deepcopy(source)
    for observation in out['projection']['observations']:
        for sample in observation['samples']:
            u,v=sample['pixel']; k=geometry.intrinsics
            if not 0<=u<k.width or not 0<=v<k.height: raise ValueError('sample pixel outside camera')
            expected=geometry.pose.unproject(u,v,sample['depth_z_m'],k)
            if max(abs(a-b) for a,b in zip(expected,sample['world_m']))>1e-8:
                raise ValueError('source point inconsistent with supplied camera pose')
            sample['source_world_m']=list(sample['world_m'])
            sample['world_m']=list(transform.point(sample['world_m'],**options))
    out['projection'].update(world_frame=target_frame,consumer_at_s=now_s,age_s=now_s-p['captured_at_s'])
    return dict(converted=True,reason=source['reason'],frame_id=p['frame_id'],clock_id=clock_id,
        source_world_frame=transform.source_frame,world_frame=target_frame,checked_at_s=now_s,
        captured_at_s=p['captured_at_s'],valid_until_s=p['valid_until_s'],transform=asdict(transform),
        reference_kind='research_coordinate_definition',physical_navigation_calibrated=False,
        camera_pose=asdict(transform.pose(geometry.pose,**options)),observation=out,
        navigation_map_update_allowed=False,flight_authorized=False,free_space_evidence=[])


def finish_transform(result,*,now_s,clock_id):
    """Caller measures conversion work, then gates publication at that final time."""
    if not number(now_s) or now_s<result['checked_at_s'] or clock_id!=result['clock_id']:
        raise ValueError('invalid final conversion clock or time')
    expired=now_s>result['valid_until_s']+1e-9
    return dict(available=not expired,reason='EXPIRED_AFTER_TRANSFORM' if expired else result['reason'],
                now_s=now_s,age_s=now_s-result['captured_at_s'],world_frame=result['world_frame'],
                result=None if expired else deepcopy(result),navigation_map_update_allowed=False,
                physical_navigation_calibrated=False,flight_authorized=False)
