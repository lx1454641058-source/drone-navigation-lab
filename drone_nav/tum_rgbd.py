"""来源：本项目原创。TUM Freiburg 3 RGB-D 只读适配；不授权飞行。

输出坐标保留 TUM 动捕世界系，不能冒称 ENU 或接入现有飞行地图。
数据、标定来源和限制见 docs/TUM_RGBD_PROTOCOL.md。
"""
from bisect import bisect_left
from dataclasses import dataclass
from math import isfinite, isnan, sqrt
from pathlib import Path, PurePosixPath
import struct

from .detection_bridge import DetectionPacket, SpatialContext, project_detections
from .pinhole import Intrinsics, PerspectiveFrame, Pose

INTRINSICS = Intrinsics(640, 480, 535.4, 539.2, 320.1, 247.6)
CLOCK = 'tum-rgbd-relative-seconds'
WORLD = 'tum_mocap_world_m'


def rows(path, width):
    """Require increasing finite timestamps; never silently sort broken records."""
    result = []
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != width:
            raise ValueError('unexpected TUM record width')
        stamp = float(parts[0])
        if not isfinite(stamp) or stamp < 0 or (result and stamp <= result[-1][0]):
            raise ValueError('timestamps must be finite and strictly increasing')
        result.append((stamp, *parts[1:]))
    if not result:
        raise ValueError('empty TUM index')
    return result


def nearest(records, stamp, max_delta=.02):
    if not isfinite(stamp) or not isfinite(max_delta) or max_delta < 0 or not records:
        raise ValueError('invalid matching request')
    index = bisect_left([r[0] for r in records], stamp)
    candidates = records[max(0, index-1):index+1]
    result = min(candidates, key=lambda r: (abs(r[0]-stamp), r[0]))
    if abs(result[0]-stamp) > max_delta + 1e-9:
        raise ValueError('no synchronized observation within tolerance')
    return result


def safe_path(root, relative):
    posix = PurePosixPath(relative)
    if (posix.is_absolute() or '..' in posix.parts or '\\' in relative
            or ':' in relative or not relative):
        raise ValueError('invalid dataset path')
    path = Path(root).joinpath(*posix.parts).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('dataset path escapes root')
    return path


def pose_from_record(record):
    values = tuple(float(v) for v in record[1:])
    if len(values) != 7 or not all(isfinite(v) for v in values):
        raise ValueError('invalid ground-truth pose')
    position = values[:3]
    x, y, z, w = values[3:]
    norm = sqrt(x*x+y*y+z*z+w*w)
    if abs(norm-1) > 1e-3:
        raise ValueError('ground-truth quaternion is not unit length')
    # Normalize only text-rounding noise. Columns of R map optical camera axes
    # (right, down, forward) to the recorded motion-capture world coordinates.
    x, y, z, w = (v/norm for v in (x, y, z, w))
    right = (1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w))
    down = (2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w))
    forward = (2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y))
    return Pose(position, right, down, forward)


@dataclass(frozen=True)
class RGBDObservation:
    frame: PerspectiveFrame
    frame_id: str
    captured_at_s: float
    depth_at_s: float
    pose_at_s: float
    depth_source: str = 'TUM registered Kinect PNG; uint16/5000; zero missing'


def load_observation(root, selection, origin_s):
    from PIL import Image
    rgb, depth, pose = (selection[k] for k in ('rgb', 'depth', 'pose'))
    stamps = (rgb[0], depth[0], pose[0])
    if (not isfinite(origin_s) or not all(isfinite(t) for t in stamps) or min(stamps) < origin_s or
            max(abs(t-rgb[0]) for t in stamps) > .02 + 1e-9):
        raise ValueError('invalid origin or unsynchronized selected records')
    with Image.open(safe_path(root, rgb[1])) as image:
        if image.size != (640, 480) or image.mode != 'RGB':
            raise ValueError('expected original 640x480 RGB PNG')
        pixels = tuple(image.getdata())
    if selection.get('depth_format', 'PNG_UINT16') == '32FC1_LE':
        raw = safe_path(root,depth[1]).read_bytes()
        if len(raw) != 640*480*4:
            raise ValueError('float32 depth length differs')
        values = struct.unpack('<307200f',raw)
        if any(not isnan(v) and (not isfinite(v) or v<0) for v in values):
            raise ValueError('unexpected negative or infinite measured depth')
        depths = tuple(v if v>0 else None for v in values)
        depth_source = 'TUM registered Kinect bag Image; optical-Z float32 metres; zero/NaN missing'
    elif selection.get('depth_format', 'PNG_UINT16') == 'PNG_UINT16':
        with Image.open(safe_path(root, depth[1])) as image:
            if image.size != (640, 480) or image.mode not in ('I;16', 'I;16B', 'I'):
                raise ValueError('expected original 16-bit depth PNG')
            values = tuple(image.getdata())
            if any(type(v) is not int or not 0 <= v <= 65535 for v in values):
                raise ValueError('depth samples outside uint16 range')
        # Zero means missing, never max range or free space. Scale already corrected upstream.
        depths = tuple(v/5000 if v else None for v in values)
        depth_source = 'TUM registered Kinect PNG; uint16/5000; zero missing'
    else:
        raise ValueError('unsupported depth storage format')
    frame = PerspectiveFrame(INTRINSICS, pose_from_record(pose), pixels, depths, 0)
    frame.validate()
    return RGBDObservation(frame, rgb[1], *(t-origin_s for t in stamps),depth_source)


def project_record(observation, record, *, completed_at_s, now_s):
    frame = observation.frame
    packet = DetectionPacket(observation.frame_id, 'tum-freiburg3-rgb', 640, 480,
        tuple(dict(id=i, group=b['group'], score=b['score'],
                   box=[b[n] for n in ('x1', 'y1', 'x2', 'y2')])
              for i, b in enumerate(record['result']['boxes'])),
        'TinyFormer:'+record['model_sha256'], observation.captured_at_s, completed_at_s, CLOCK)
    context = SpatialContext(observation.frame_id, packet.camera_id, CLOCK,
        observation.depth_at_s, observation.pose_at_s, frame.intrinsics, frame.pose,
        frame.depth_z_m, True, 'camera_optical_axis_z_m',
        observation.depth_source,
        'TUM motion-capture nearest pose; no interpolation', 'TUM Freiburg3 RGB undistorted')
    result = project_detections(packet, context, now_s=now_s, now_clock_id=CLOCK)
    # The old bridge uses an ENU label for its simulation callers. No downstream
    # map sees that label here: publish the actual measured coordinate system.
    result['world_frame'] = WORLD if result['status'] != 'CONTEXT_REJECTED' else None
    return dict(projection=result, reason=('VISUAL_CONTEXT_HOLD' if result['status']=='CONTEXT_REJECTED'
        else 'VISUAL_TARGET_HOLD' if packet.detections else 'NO_DETECTION_REQUIRES_GEOMETRY'),
        flight_authorized=False, navigation_frame_alignment_available=False)
