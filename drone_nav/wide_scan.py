"""来源：本项目原创。扩大导航扫描视场，保持焦距、真实采集时间和原控制预算。"""
from dataclasses import replace
from math import atan2,cos,sin,pi
from .exploration import scan_poses
from .pinhole import Intrinsics,Pose
from .sampling_motion import SamplingView


class WideScanMixin:
    def scan(self,tick,aim):
        if not self.spec.get('wide_scan',False):return super().scan(tick,aim)
        self._navigation_aim=tuple(aim)
        self.retire_expired_views()
        # fx/fy 仍为 25 像素，扩大画面不粗化原中心区域的采样间距。
        k=Intrinsics(80,60,25,25,39.5,29.5)
        frames=[]
        upper=not (self.probes and self.probes[-1]['status']=='RETURN_CONFIRMED')
        for group in range(2 if upper else 1):
            records=[]
            for i in range(8):
                state=self.state()
                if group==0:
                    def target(xyz,i=i):
                        if self.spec.get('level_scan',False):
                            angle=i*pi/4
                            return Pose.look_at(xyz,(xyz[0]+8*cos(angle),xyz[1]+8*sin(angle),xyz[2]))
                        return scan_poses(xyz,aim)[i]
                else:
                    def target(xyz,i=i):
                        angle=atan2(aim[1]+.5-xyz[1],aim[0]+.5-xyz[0])+i*pi/4
                        return Pose.look_at(xyz,(xyz[0]+8*cos(angle),xyz[1]+8*sin(angle),xyz[2]+3))
                frame=self.capture(k,tick,target)
                if frame.intrinsics!=k or frame.tick!=tick:
                    raise ValueError('wide scan calibration/sequence mismatch')
                frame.validate();frames.append(frame);records.append((state,frame))
                self.hold(.1)
            self.hold(.1);available=self.state()['time_s']
            for state,frame in records:
                sequence=self.sampling_history.grid.last_tick+1
                self.sampling_history.add(SamplingView(replace(frame,tick=sequence),
                    ('upper-' if group else 'nav-')+str(sequence),'camera','physics-seconds',
                    state['time_s'],available),now_s=available)
                self.source_frames.append(dict(navigation_tick=tick,ledger_tick=sequence,
                    captured_at_s=state['time_s'],available_at_s=available))
        return frames
