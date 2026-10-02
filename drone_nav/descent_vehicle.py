"""来源：本项目原创。降落试验的传感器故障注入与几何间距探针。"""

from dataclasses import asdict,replace
from math import isfinite
from .physical_vehicle import PhysicalVehicle


class DescentVehicle(PhysicalVehicle):
    def __init__(self,*args,fault=None,fault_after_s=4.0,**kwargs):
        if fault not in (None,'depth','semantic','contact'):
            raise ValueError('unknown descent fault')
        if type(fault_after_s) not in (int,float) or not isfinite(fault_after_s) or fault_after_s<0:
            raise ValueError('fault time must be finite and nonnegative')
        super().__init__(*args,**kwargs)
        self.fault,self.fault_after_s=fault,fault_after_s
        self.descent_start_s=None;self.contact_readings=[]

    def fault_active(self):
        return self.descent_start_s is not None and self.state()['time_s']-self.descent_start_s>=self.fault_after_s

    def capture(self,k,tick,aim=None):
        frame=super().capture(k,tick,aim)
        if self.fault_active():
            if self.fault=='depth':frame=replace(frame,depth_z_m=(None,)*len(frame.depth_z_m))
            if self.fault=='semantic':frame=replace(frame,rgb=((30,100,180),)*len(frame.rgb))
        # 保存的是实际交给算法的像素，而不是注入之前的原图。
        self.frames[-1]['frame']=asdict(frame)
        return frame

    def contact_gap(self):
        """模拟几何间距探针，不是接触力或真实硬件开关。保留真值用于独立评价。"""
        gap=self.physics.ground_distance()
        value=None if self.fault=='contact' and self.fault_active() else gap
        self.contact_readings.append(dict(step=self.steps,time_s=self.state()['time_s'],value=value,truth_gap_m=gap))
        return value
