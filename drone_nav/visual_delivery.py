"""来源：本项目原创。透视导航到达后，再用当前图像检查固定取餐点。"""

from dataclasses import asdict
from math import isfinite

from .localization import LocalizationBudget
from .perspective_landing import LandingConfig,inspect_landing,landing_preview
from .pinhole import Intrinsics,Pose
from .timed_navigation import navigate


def run_visual_delivery(camera,model,start,goal,*,reference_z_m=0.0,landing_config=None,
                        landing_intrinsics=None,previews=True):
    """当前仅支持理想位姿；相机协议增加 capture_landing，场景几何不会传入本层。"""
    config = landing_config or LandingConfig()
    k = landing_intrinsics or Intrinsics(96,72,45,45,47.5,35.5)
    if type(reference_z_m) not in (int,float) or not isfinite(reference_z_m) or reference_z_m>=3.5-config.elevation_tolerance_m:
        raise ValueError('reference ground height must be finite and below the inspection camera')
    nav = navigate(camera,20,16,start,goal,localization=LocalizationBudget(),
                   policy_mode='reachable',previews=previews)
    result = {'navigation':nav,'landing_config':asdict(config),'landing_intrinsics':asdict(k),
              'reference_z_m':reference_z_m,'model_digest':model.digest,'landing':None,
              'landing_labels':None,'landing_images':None,'terminal_state':'NAVIGATION_HOLD',
              'reason':'导航未确认到达，未进入降落检查','limitations':'当前使用精确位姿和合成颜色模型；规则通过不表示实际降落或外卖送达。'}
    if nav['terminal_state']!='ARRIVED_WAYPOINT':
        return result
    position = nav['trace'][-1]['position']
    xyz = (position[0]+.5,position[1]+.5,nav['altitude_m'])
    target = (goal[0]+.5,goal[1]+.5,reference_z_m)
    tick = nav['observations']
    pose = Pose.look_at(xyz,target)
    try:
        frame = camera.capture_landing(pose,k,tick)
        if frame.pose!=pose or frame.intrinsics!=k or frame.tick!=tick:
            raise ValueError('landing observation differs from requested pose/time/calibration')
        evidence,labels = inspect_landing(frame,model,target,expected_tick=tick,expected_position=xyz,config=config)
        result.update(landing=evidence,landing_labels=labels,
                      landing_images=landing_preview(frame,evidence,labels) if previews else None,
                      terminal_state='READY_TO_LAND' if evidence['accepted'] else 'LANDING_REJECTED',reason=evidence['reason'])
    except (ValueError,TypeError,OverflowError) as exc:
        result.update(terminal_state='LANDING_SENSOR_HOLD',reason='降落观测无效：'+str(exc))
    return result
