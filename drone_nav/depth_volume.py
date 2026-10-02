"""来源：本项目原创。同参考坐标下的密集深度体积查询；不证明空闲。"""
from dataclasses import dataclass
from itertools import product
from math import ceil,floor,sqrt

from .detection_bridge import identifier,number
from .frame_transform import RigidFrameTransform
from .framed_surface_map import FramedSurfaceMap
from .packed_rgbd import PackedObservation,PackedFrame
from .pinhole import Intrinsics,Pose


@dataclass(frozen=True)
class QueryVolume:
    """Axis-aligned research box, already including any caller-chosen margins."""
    name: str
    world_frame: str
    lower: tuple
    upper: tuple

    def __post_init__(self):
        if not all(identifier(v) for v in (self.name,self.world_frame)):
            raise ValueError('volume identity required')
        if any(type(v) is not tuple or len(v)!=3 or not all(number(n) and abs(n)<=10000 for n in v)
               for v in (self.lower,self.upper)) or any(a>=b for a,b in zip(self.lower,self.upper)):
            raise ValueError('immutable ordered finite box bounds required')

    def contains(self,point):
        return all(a-1e-9<=v<=b+1e-9 for a,v,b in zip(self.lower,point,self.upper))


class DepthVolumeInspector:
    """Bind registered depth to an already validated semantic frame and reference.

    Constructor and queries are research-only. Full projected windows are read;
    an absence of measured endpoints inside a box never certifies free volume.
    """
    def __init__(self,observation,finished,transform,*,camera_id,clock_id,expected_intrinsics):
        if type(observation) is not PackedObservation or type(observation.frame) is not PackedFrame:
            raise ValueError('validated packed depth observation required')
        if type(transform) is not RigidFrameTransform: raise ValueError('validated frame transform required')
        if type(expected_intrinsics) is not Intrinsics or observation.frame.intrinsics!=expected_intrinsics:
            raise ValueError('depth intrinsics differ from explicitly selected calibration')
        receiver=FramedSurfaceMap(transform,camera_id=camera_id,clock_id=clock_id)
        now=finished.get('now_s') if type(finished) is dict else None
        accepted=receiver.ingest(finished,now_s=now,clock_id=clock_id)
        if not accepted['accepted']: raise ValueError('semantic context rejected: '+accepted['reason'])
        mapped=finished['result']; p=mapped['observation']['projection']
        times=(observation.captured_at_s,observation.depth_at_s,observation.pose_at_s)
        if (observation.frame_id!=p['frame_id'] or observation.captured_at_s!=p['captured_at_s']
                or not identifier(observation.depth_source) or observation.depth_source!=p.get('sources',{}).get('depth')
                or any(not number(t) or t<0 or t>now for t in times)
                or max(abs(t-times[0]) for t in times)>.02+1e-9):
            raise ValueError('depth/pose capture identities or timestamps differ')
        frame=observation.frame
        pose=transform.pose(frame.pose,source_frame=transform.source_frame,target_frame=transform.target_frame)
        for key in ('position','right','down','forward'):
            if max(abs(a-b) for a,b in zip(getattr(pose,key),mapped['camera_pose'][key]))>1e-8:
                raise ValueError('depth camera pose differs from converted camera')
        semantic=[]
        for entry in p['observations']:
            for sample in entry['samples']:
                u,v=sample['pixel']; k=frame.intrinsics
                if not (0<=u<k.width and 0<=v<k.height): raise ValueError('semantic pixel outside depth')
                depth=frame.depth_z_m[v*k.width+u]
                if depth!=sample['depth_z_m']: raise ValueError('semantic/depth measurement differs')
                point=pose.unproject(u,v,depth,k)
                if max(abs(a-b) for a,b in zip(point,sample['world_m']))>1e-8:
                    raise ValueError('semantic point differs from bound depth geometry')
                semantic.append((entry['detection']['id'],entry['detection']['group'],u,v,*point))
        self.intrinsics=frame.intrinsics; self.pose=pose; self.depths=frame.depth_z_m
        self.semantic=tuple(semantic); self.world_frame=transform.target_frame
        self.reference_fingerprint=receiver.fingerprint; self.frame_id=observation.frame_id
        self.camera_id=camera_id; self.clock_id=clock_id; self.available_at_s=now
        # Depth or pose may precede RGB. The earliest component must still be live.
        self.valid_until_s=min(p['valid_until_s'],min(times)+.5)
        self.captured_at_s=times[0]; self.depth_source=observation.depth_source

    def inspect(self,volume,*,now_s,clock_id,depth_error_m=.05,max_range_m=30.):
        if type(volume) is not QueryVolume or volume.world_frame!=self.world_frame:
            raise ValueError('query volume frame differs')
        if clock_id!=self.clock_id or not number(now_s) or now_s<self.available_at_s:
            raise ValueError('query clock differs or precedes data availability')
        if not number(depth_error_m) or not 0<=depth_error_m<=1 or not number(max_range_m) or not 0<max_range_m<=100:
            raise ValueError('finite depth margin/range required')
        result=dict(volume=volume.name,world_frame=self.world_frame,frame_id=self.frame_id,camera_id=self.camera_id,
            clock_id=self.clock_id,reference_fingerprint=self.reference_fingerprint,checked_at_s=now_s,
            valid_until_s=self.valid_until_s,captured_at_s=self.captured_at_s,depth_source=self.depth_source,
            sampling_model='point_samples',flight_authorized=False,free_volume_proven=False,
            navigation_map_update_allowed=False,depth_error_m=depth_error_m,max_range_m=max_range_m,
            reasons=[],checked_pixels=0,missing_pixels=0,out_of_range_pixels=0,
            foreground_pixels=0,inside_surface_pixels=0,semantic_surface_samples=0,witnesses=[])
        if now_s>self.valid_until_s+1e-9:
            return dict(result,status='CONTEXT_REJECTED',reasons=['DEPTH_OR_POSE_EXPIRED'])
        corners=[self.pose.project(p,self.intrinsics) for p in product(*zip(volume.lower,volume.upper))]
        if any(p is None for p in corners):
            return dict(result,status='INSUFFICIENT_VIEW',reasons=['VOLUME_NOT_FULLY_IN_FRONT'])
        k=self.intrinsics
        window=[ceil(min(p[0] for p in corners)-.5-1e-9),ceil(min(p[1] for p in corners)-.5-1e-9),
                floor(max(p[0] for p in corners)+.5+1e-9),floor(max(p[1] for p in corners)+.5+1e-9)]
        far=max(p[2] for p in corners)
        result.update(window=window,far_z_m=far,pixel_pitch_at_far_m=[far/k.fx,far/k.fy])
        x0,y0,x1,y1=window
        if x0<0 or y0<0 or x1>=k.width or y1>=k.height:
            return dict(result,status='INSUFFICIENT_VIEW',reasons=['INCOMPLETE_VOLUME_VIEW'])
        for v in range(y0,y1+1):
            for u in range(x0,x1+1):
                result['checked_pixels']+=1; z=self.depths[v*k.width+u]
                if z is None:
                    result['missing_pixels']+=1; continue
                ray=k.ray(u,v)
                if z*sqrt(sum(a*a for a in ray))>max_range_m:
                    result['out_of_range_pixels']+=1; continue
                if z-depth_error_m<=far+1e-9: result['foreground_pixels']+=1
                point=self.pose.unproject(u,v,z,k)
                if volume.contains(point):
                    result['inside_surface_pixels']+=1
                    if len(result['witnesses'])<8: result['witnesses'].append(dict(pixel=[u,v],depth_z_m=z,world_m=list(point)))
        # Semantic samples are a subset of measured depth endpoints, not object extents.
        result['semantic_surface_samples']=sum(volume.contains(sample[4:]) for sample in self.semantic)
        if result['inside_surface_pixels']: result['reasons'].append('MEASURED_SURFACE_IN_VOLUME')
        if result['foreground_pixels']: result['reasons'].append('FOREGROUND_OR_DEPTH_MARGIN')
        if result['missing_pixels']: result['reasons'].append('MISSING_DEPTH')
        if result['out_of_range_pixels']: result['reasons'].append('OUT_OF_RANGE')
        result['reasons'].append('PIXEL_GAPS_AND_ERROR_BOUNDS_UNVERIFIED')
        result['status']=('MEASURED_SURFACE' if result['inside_surface_pixels'] else 'INSUFFICIENT_DEPTH'
            if result['missing_pixels'] or result['out_of_range_pixels'] else 'FOREGROUND_OR_OCCLUSION'
            if result['foreground_pixels'] else 'SAMPLED_BEYOND_VOLUME')
        return result
