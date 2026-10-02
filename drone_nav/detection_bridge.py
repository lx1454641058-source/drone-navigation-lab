"""来源：本项目原创。二维检测与同步 RGB-D 的内部研究接口，不授权飞行。

像素中心为整数，窗口为半开区间。深度必须为已配准至 RGB 的相机 z 轴米数。
框内点可能属于背景/遮挡物；本模块不估计物体中心、完整范围或空闲空间。
"""
from dataclasses import dataclass
from math import ceil,isfinite,sqrt

from .pinhole import Intrinsics,Pose


def number(value):
    return type(value) in (int,float) and isfinite(value)


def identifier(value):
    return isinstance(value,str) and bool(value.strip())


@dataclass(frozen=True)
class DetectionPacket:
    frame_id: str
    camera_id: str
    width: int
    height: int
    detections: tuple[dict,...]
    source: str
    captured_at_s: float|None=None
    completed_at_s: float|None=None
    clock_id: str|None=None

    def validate(self):
        if (not all(identifier(v) for v in (self.frame_id,self.camera_id,self.source)) or
            any(type(v) is not int or v<1 for v in (self.width,self.height)) or self.width*self.height>16_000_000):
            raise ValueError('invalid detection frame identity or size')
        if self.clock_id is not None and not identifier(self.clock_id):raise ValueError('invalid clock identity')
        if any(v is not None and (not number(v) or v<0) for v in (self.captured_at_s,self.completed_at_s)):
            raise ValueError('invalid detection timestamp')
        if not isinstance(self.detections,(tuple,list)) or len(self.detections)>30000:raise ValueError('invalid detection list')
        ids=set()
        for d in self.detections:
            if not isinstance(d,dict) or type(d.get('id')) is not int or d['id']<0 or d['id'] in ids:
                raise ValueError('invalid or duplicate detection id')
            ids.add(d['id']);b=d.get('box')
            if (d.get('group') not in ('person','vehicle') or not number(d.get('score')) or not 0<=d['score']<=1 or
                not isinstance(b,(tuple,list)) or len(b)!=4 or not all(number(v) for v in b) or
                not 0<=b[0]<b[2]<=self.width or not 0<=b[1]<b[3]<=self.height):
                raise ValueError('invalid detection box, group or score')


@dataclass(frozen=True)
class SpatialContext:
    frame_id: str
    camera_id: str
    clock_id: str
    depth_at_s: float
    pose_at_s: float
    intrinsics: Intrinsics
    pose: Pose
    depth_z_m: tuple[float|None,...]
    registered_to_rgb: bool
    depth_convention: str
    depth_source: str
    pose_source: str
    calibration_id: str

    def validate(self):
        if not all(identifier(v) for v in (self.frame_id,self.camera_id,self.clock_id,self.depth_source,self.pose_source,self.calibration_id)):
            raise ValueError('invalid spatial source identity')
        if not isinstance(self.intrinsics,Intrinsics) or not isinstance(self.pose,Pose):raise ValueError('invalid spatial geometry')
        if any(not number(v) or v<0 for v in (self.depth_at_s,self.pose_at_s)):raise ValueError('invalid context time')
        if type(self.registered_to_rgb) is not bool or not isinstance(self.depth_convention,str):raise ValueError('invalid depth metadata')
        if len(self.depth_z_m)!=self.intrinsics.width*self.intrinsics.height:raise ValueError('depth grid shape differs')
        if any(z is not None and (not number(z) or z<=0) for z in self.depth_z_m):raise ValueError('invalid depth; use None for missing')


@dataclass(frozen=True)
class BridgeConfig:
    max_age_s: float=.5
    max_sync_error_s: float=.02
    max_range_m: float=30
    samples_per_axis: int=5

    def __post_init__(self):
        if any(not number(v) or v<=0 for v in (self.max_age_s,self.max_range_m)) or not number(self.max_sync_error_s) or self.max_sync_error_s<0:
            raise ValueError('invalid bridge limits')
        if type(self.samples_per_axis) is not int or not 2<=self.samples_per_axis<=16:raise ValueError('invalid sampling budget')


def sample_pixels(box,count):
    """只取窗口内整数像素中心；极窄窗口可能不含任何像素中心。"""
    x0,y0,x1,y1=ceil(box[0]),ceil(box[1]),ceil(box[2])-1,ceil(box[3])-1
    if x1<x0 or y1<y0:return []
    xs=sorted({round(x0+(x1-x0)*i/(count-1)) for i in range(count)})
    ys=sorted({round(y0+(y1-y0)*i/(count-1)) for i in range(count)})
    return [(x,y) for y in ys for x in xs]


def project_detections(packet:DetectionPacket,context:SpatialContext|None,*,now_s:float,now_clock_id:str,config=None):
    packet.validate();c=config or BridgeConfig()
    if not number(now_s) or now_s<0 or not identifier(now_clock_id):raise ValueError('invalid consumer time')
    reasons=[];age=None
    if packet.captured_at_s is None or packet.completed_at_s is None or packet.clock_id is None:
        reasons.append('MISSING_TIMING')
    elif packet.clock_id!=now_clock_id:
        reasons.append('CONSUMER_CLOCK_MISMATCH')
    else:
        age=now_s-packet.captured_at_s
        if packet.completed_at_s<packet.captured_at_s or packet.completed_at_s>now_s or age<0:reasons.append('INVALID_TIME_ORDER')
        if age>c.max_age_s+1e-9:reasons.append('STALE_CAPTURE')
    if context is None:reasons.append('MISSING_SPATIAL_CONTEXT')
    else:
        context.validate()
        if (context.frame_id,context.camera_id)!=(packet.frame_id,packet.camera_id):reasons.append('FRAME_IDENTITY_MISMATCH')
        if context.clock_id!=packet.clock_id:reasons.append('CLOCK_MISMATCH')
        if (context.intrinsics.width,context.intrinsics.height)!=(packet.width,packet.height):reasons.append('IMAGE_SIZE_MISMATCH')
        if not context.registered_to_rgb:reasons.append('UNREGISTERED_DEPTH')
        if context.depth_convention!='camera_optical_axis_z_m':reasons.append('DEPTH_CONVENTION_MISMATCH')
        if context.clock_id==now_clock_id and (context.depth_at_s>now_s or context.pose_at_s>now_s):reasons.append('FUTURE_CONTEXT')
        if packet.captured_at_s is not None and max(abs(context.depth_at_s-packet.captured_at_s),abs(context.pose_at_s-packet.captured_at_s))>c.max_sync_error_s+1e-9:
            reasons.append('UNSYNCHRONIZED_CONTEXT')
    observations=[]
    for d in packet.detections:
        samples=[];missing=outside=0;pixels=[]
        if not reasons:
            pixels=sample_pixels(d['box'],c.samples_per_axis)
            for u,v in pixels:
                z=context.depth_z_m[v*packet.width+u]
                if z is None:missing+=1;continue
                ray=context.intrinsics.ray(u,v)
                if z*sqrt(sum(a*a for a in ray))>c.max_range_m:outside+=1;continue
                world=context.pose.unproject(u,v,z,context.intrinsics)
                samples.append(dict(pixel=[u,v],depth_z_m=z,world_m=list(world)))
        local=list(reasons)
        if not reasons:
            if not pixels:local.append('NO_PIXEL_CENTERS')
            elif not samples:local.append('NO_VALID_DEPTH_SAMPLES')
            elif missing or outside:local.append('PARTIAL_DEPTH_SUPPORT')
        observations.append(dict(detection=dict(d),status='SURFACE_SAMPLES' if samples else 'IMAGE_ONLY',
            reasons=local,samples=samples,sampled_pixels=len(pixels),missing_depth=missing,out_of_range=outside,
            association='unverified_box_surface',object_extent_known=False))
    # 所有有效三维点仍是框内可见表面样本，不是飞行授权或无障碍证明。
    return dict(frame_id=packet.frame_id,camera_id=packet.camera_id,detector_source=packet.source,
        captured_at_s=packet.captured_at_s,completed_at_s=packet.completed_at_s,consumer_at_s=now_s,
        age_s=age,valid_until_s=None if packet.captured_at_s is None else packet.captured_at_s+c.max_age_s,
        clock_id=packet.clock_id,consumer_clock_id=now_clock_id,world_frame='east_north_up_m' if context is not None and not reasons else None,
        sources=None if context is None else dict(depth=context.depth_source,pose=context.pose_source,calibration=context.calibration_id),
        status='CONTEXT_REJECTED' if reasons else 'SAMPLES_AVAILABLE' if any(o['samples'] for o in observations) else 'NO_SPATIAL_SAMPLES',
        reasons=reasons,observations=observations,free_space_evidence=[],flight_authorized=False)
