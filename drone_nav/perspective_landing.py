"""来源：本项目原创。透视 RGB-D 的降落点证据，只读取像素、标定与任务坐标。"""

from collections import Counter
from dataclasses import asdict,dataclass
from math import atan,degrees,hypot,isfinite,sqrt

from .imaging import CLASSES,DISPLAY_COLORS,png_url
from .pinhole import finite_vector
from .surface_geometry import fit_plane


@dataclass(frozen=True)
class LandingConfig:
    radius_m: float = 2.0
    max_slope_deg: float = 5.0
    max_roughness_m: float = .08
    max_residual_m: float = .15
    elevation_tolerance_m: float = .2
    min_samples: int = 25

    def __post_init__(self):
        for value in (self.radius_m,self.max_slope_deg,self.max_roughness_m,self.max_residual_m,self.elevation_tolerance_m):
            if type(value) not in (int,float) or not isfinite(value) or value<=0:
                raise ValueError('landing limits must be finite and positive')
        if self.max_slope_deg>=45 or type(self.min_samples) is not int or self.min_samples<3:
            raise ValueError('invalid slope limit or sample count')


def inspect_landing(frame,model,target,*,expected_tick,expected_position,config=None):
    """target 是任务指定的 (x,y,参考地面高度)，不是从场景真值查询的表面。"""
    config = config or LandingConfig()
    frame.validate()
    if (not finite_vector(target,3) or not finite_vector(expected_position,3)
            or type(expected_tick) is not int or frame.tick!=expected_tick
            or any(abs(a-b)>1e-8 for a,b in zip(frame.pose.position,expected_position))):
        raise ValueError('stale landing frame or inconsistent target/camera position')
    if frame.pose.position[2]<=target[2]+config.elevation_tolerance_m:
        raise ValueError('landing camera must be above the reference surface')
    k,pose,radius = frame.intrinsics,frame.pose,config.radius_m
    labels = model.predict_image(frame.rgb)
    evidence = {'source':'perspective_rgbd_plane_fit','config':asdict(config),'target':target,
                'geometry_accepted':False,'semantic_accepted':False,'accepted':False,
                'slope_deg':None,'roughness_m':None,'max_residual_m':None,'center_height_m':None,
                'surface':'unknown','class_counts':{},'sample_count':0,'missing_depth':0,
                'coverage_complete':False,'roi_pixels':[],'reason_codes':[],'reason':''}
    # 检查参考高度上下容差内的整个外接方柱，而非只确认中心在图像里。
    projected = [pose.project((target[0]+sx*radius,target[1]+sy*radius,
                               target[2]+sz*config.elevation_tolerance_m),k)
                 for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)]
    covered = all(p is not None and -.5<=p[0]<k.width-.5 and -.5<=p[1]<k.height-.5 for p in projected)
    evidence['coverage_complete'] = covered
    samples,selected_labels = [],[]
    for i,depth in enumerate(frame.depth_z_m):
        direction = pose.rotate(k.ray(i%k.width,i//k.width))
        t = (target[2]-pose.position[2])/direction[2] if abs(direction[2])>1e-12 else -1
        reference_xy = tuple(pose.position[j]+t*direction[j] for j in range(2))
        in_reference = t>0 and hypot(reference_xy[0]-target[0],reference_xy[1]-target[1])<=radius
        point = pose.unproject(i%k.width,i//k.width,depth,k) if depth is not None else None
        in_actual = point is not None and hypot(point[0]-target[0],point[1]-target[1])<=radius
        if not (in_reference or in_actual):
            continue
        evidence['roi_pixels'].append(i)
        selected_labels.append(labels[i])
        if point is None:
            evidence['missing_depth'] += 1
        else:
            samples.append((point[0]-target[0],point[1]-target[1],point[2]))
    counts = Counter(selected_labels)
    evidence['sample_count'] = len(samples)
    evidence['class_counts'] = {CLASSES[label] if label>=0 else 'unknown':count for label,count in sorted(counts.items())}
    if counts:
        dominant = counts.most_common(1)[0][0]
        evidence['surface'] = CLASSES[dominant] if dominant>=0 else 'unknown'
    evidence['semantic_accepted'] = bool(selected_labels) and all(label==0 for label in selected_labels)
    codes = evidence['reason_codes']
    if not covered: codes.append('INCOMPLETE_COVERAGE')
    if evidence['missing_depth']: codes.append('MISSING_DEPTH')
    if len(samples)<config.min_samples: codes.append('INSUFFICIENT_POINTS')
    if not codes:
        try:
            a,b,c = fit_plane(samples)
            residuals = [abs(z-a*x-b*y-c) for x,y,z in samples]
            slope,roughness,peak = degrees(atan(hypot(a,b))),sqrt(sum(r*r for r in residuals)/len(residuals)),max(residuals)
            evidence.update(slope_deg=slope,roughness_m=roughness,max_residual_m=peak,center_height_m=c)
            if slope>config.max_slope_deg: codes.append('EXCESSIVE_SLOPE')
            if roughness>config.max_roughness_m or peak>config.max_residual_m: codes.append('UNEVEN_SURFACE')
            if abs(c-target[2])>config.elevation_tolerance_m: codes.append('UNEXPECTED_ELEVATION')
        except ValueError:
            codes.append('DEGENERATE_PLANE')
    evidence['geometry_accepted'] = not codes
    if not evidence['semantic_accepted']: codes.append('NON_PAVED_OR_UNKNOWN')
    evidence['accepted'] = not codes
    reasons = {'INCOMPLETE_COVERAGE':'候选区域没有完整进入相机视野','MISSING_DEPTH':'候选区域存在缺失深度',
               'INSUFFICIENT_POINTS':'有效空间点不足','EXCESSIVE_SLOPE':'坡度超过配置阈值',
               'UNEVEN_SURFACE':'存在凸起或表面不平整','UNEXPECTED_ELEVATION':'地面高度与任务参考值不符',
               'DEGENERATE_PLANE':'空间点不能支持平面拟合','NON_PAVED_OR_UNKNOWN':'包含非铺装表面或未知类别'}
    evidence['reason'] = '；'.join(reasons[c] for c in codes) if codes else '图像与几何规则通过；尚未下降或触地'
    return evidence,labels


def landing_preview(frame,evidence,labels):
    selected = set(evidence['roi_pixels'])
    mask = [DISPLAY_COLORS[label] for label in labels]
    roi = [rgb if i in selected else tuple(v//4 for v in rgb) for i,rgb in enumerate(frame.rgb)]
    depth = [(0,0,0) if z is None else (min(255,int(z/6*255)),)*3 for z in frame.depth_z_m]
    k = frame.intrinsics
    return {key:png_url(k.width,k.height,pixels) for key,pixels in
            (('rgb',frame.rgb),('mask',mask),('roi',roi),('depth',depth))}
