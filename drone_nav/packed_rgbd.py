"""来源：本项目原创。不可变字节 RGB 输入，保留原始浮点深度及原模型张量。"""
from array import array
from dataclasses import dataclass
import gzip
from math import isfinite, isnan
import struct
import sys
from time import perf_counter

from .pinhole import Intrinsics, Pose
from .realvision import dump, sha
from .semantic_sensor import TinyFormerSession
from .tum_rgbd import INTRINSICS, pose_from_record, safe_path
from tools.tinyformer_probe import f32, MODEL_SHA


def normalization_tables():
    tables=[]
    for mean,std in zip((.485,.456,.406),(.229,.224,.225)):
        mean,std=f32(mean),f32(std)
        tables.append(tuple(f32(f32(f32(v/255)-mean)/std) for v in range(256)))
    return tuple(tables)


TABLES=normalization_tables()


def prepare_packed(image,window):
    """Same crop, Pillow bilinear resize and three float32 rounding stages as reference."""
    from PIL import Image
    x,y,w,h=(window[k] for k in ('x0','y0','width','height'))
    if any(type(v) is not int for v in (x,y,w,h)) or min(x,y)<0 or min(w,h)<=0 or x+w>image.width or y+h>image.height:
        raise ValueError('crop outside image')
    resized=image.convert('RGB').crop((x,y,x+w,y+h)).resize((640,640),Image.Resampling.BILINEAR)
    result=b''.join(resized.getchannel(c).point(TABLES[c],mode='F').tobytes() for c in range(3))
    if sys.byteorder!='little':
        values=array('f'); values.frombytes(result); values.byteswap(); result=values.tobytes()
    return result


@dataclass(frozen=True)
class PackedFrame:
    intrinsics: Intrinsics
    pose: Pose
    rgb_bytes: bytes
    depth_z_m: tuple

    def __post_init__(self):
        if type(self.intrinsics) is not Intrinsics or type(self.pose) is not Pose:
            raise ValueError('expected validated immutable camera geometry')
        size=self.intrinsics.width*self.intrinsics.height
        # Each element of immutable bytes is an unsigned 8-bit value by definition.
        # This enforces the same RGB value/shape contract without making ~1M Python calls.
        if type(self.rgb_bytes) is not bytes or len(self.rgb_bytes)!=size*3:
            raise ValueError('expected immutable packed RGB bytes')
        if type(self.depth_z_m) is not tuple or len(self.depth_z_m)!=size:
            raise ValueError('expected immutable depth grid')
        if any(d is not None and (type(d) not in (float,int) or not isfinite(d) or d<=0) for d in self.depth_z_m):
            raise ValueError('invalid positive optical-Z depth')


@dataclass(frozen=True)
class PackedObservation:
    frame: PackedFrame
    frame_id: str
    captured_at_s: float
    depth_at_s: float
    pose_at_s: float
    depth_source: str


def load_packed(root,selection,origin_s):
    from PIL import Image
    rgb,depth,pose=(selection[k] for k in ('rgb','depth','pose'))
    stamps=(rgb[0],depth[0],pose[0])
    if (not isfinite(origin_s) or not all(isfinite(t) for t in stamps) or min(stamps)<origin_s
            or max(abs(t-rgb[0]) for t in stamps)>.02+1e-9):
        raise ValueError('invalid origin or unsynchronized selected records')
    if selection.get('depth_format')!='32FC1_LE':
        raise ValueError('packed TUM adapter requires registered float32 depth')
    with Image.open(safe_path(root,rgb[1])) as image:
        if image.size!=(640,480) or image.mode!='RGB':
            raise ValueError('expected original 640x480 RGB')
        pixels=image.tobytes()
    raw=safe_path(root,depth[1]).read_bytes()
    if len(raw)!=640*480*4:
        raise ValueError('float32 depth length differs')
    values=struct.unpack('<307200f',raw)
    # Reject bad measurements before mapping sensor-defined missing values to None.
    if any(not isnan(v) and (not isfinite(v) or v<0) for v in values):
        raise ValueError('unexpected negative or infinite measured depth')
    depths=tuple(v if v>0 else None for v in values)
    frame=PackedFrame(INTRINSICS,pose_from_record(pose),pixels,depths)
    return PackedObservation(frame,rgb[1],*(t-origin_s for t in stamps),
        'TUM registered Kinect bag Image; optical-Z float32 metres; zero/NaN missing')


class PackedTinyFormerSession(TinyFormerSession):
    def detect_packed(self,frame):
        from PIL import Image
        if self.replay or type(frame) is not PackedFrame:
            raise ValueError('validated immutable packed frame required')
        started=perf_counter()
        index=len(self.records); directory=self.output/f'frame-{index:04d}'
        directory.mkdir(exist_ok=False)
        k=frame.intrinsics
        image=Image.frombytes('RGB',(k.width,k.height),frame.rgb_bytes)
        image.save(directory/'input.png')
        window=dict(index=0,x0=0,y0=0,width=k.width,height=k.height)
        tensor=prepare_packed(image,window)
        prepared=perf_counter()
        (directory/'input.gz').write_bytes(gzip.compress(tensor,compresslevel=1,mtime=0))
        archived=perf_counter()
        result=self._request(dict(directory=str(directory),window=window))
        elapsed=perf_counter()-started
        record=dict(index=index,directory=directory.name,window=window,
            input_rgb_sha256=sha(directory/'input.png'),elapsed_s=elapsed,
            prepare_s=prepared-started,input_archive_s=archived-prepared,
            model_sha256=MODEL_SHA,result=result)
        self.records.append(record); dump(directory/'call.json',record)
        return record


def handoff_decision(result,*,consumer_at_s,clock_id):
    """Recheck age after projection work; crossing the deadline cannot restore permission."""
    projection=result['projection']
    previous=projection['consumer_at_s']; deadline=projection['valid_until_s']
    if (type(consumer_at_s) not in (float,int) or not isfinite(consumer_at_s)
            or consumer_at_s<previous or clock_id!=projection['consumer_clock_id']):
        raise ValueError('invalid handoff time or clock')
    expired=deadline is None or consumer_at_s>deadline+1e-9
    reason='VISUAL_CONTEXT_HOLD' if expired else result['reason']
    return dict(reason=reason,consumer_at_s=consumer_at_s,expired_at_handoff=expired,
        permit_geometry_check=reason=='NO_DETECTION_REQUIRES_GEOMETRY',
        flight_authorized=False,navigation_frame_alignment_available=False)
