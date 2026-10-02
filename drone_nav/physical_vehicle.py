"""来源：本项目原创。物理执行器与理想随体相机；世界只用于物理/成像。"""

from dataclasses import asdict,dataclass
from math import floor,isfinite,sqrt
import xml.etree.ElementTree as ET

from .exploration import scan_poses
from .flight_control import control
from .localization import segment_box_clearance
from .native_physics import DEFAULT_DLL,MODEL,QuadrotorPhysics
from .pinhole import Intrinsics,Pose
from .raycast import Box,Surface,render


def rotate_vector(v,q):
    w,x,y,z=q
    a,b,c=v
    return ((1-2*(y*y+z*z))*a+2*(x*y-w*z)*b+2*(x*z+w*y)*c,
            2*(x*y+w*z)*a+(1-2*(x*x+z*z))*b+2*(y*z-w*x)*c,
            2*(x*z-w*y)*a+2*(y*z+w*x)*b+(1-2*(x*x+y*y))*c)


def world_xml(world):
    """本阶段仅接收一个 z=0 大平面和静态长方体；不悄悄忽略斜坡或多地面。"""
    planes=[s for s in world.surfaces if isinstance(s,Surface)]
    if len(planes)!=1 or any(v!=0 for v in (planes[0].z0,planes[0].slope_x,planes[0].slope_y)):
        raise ValueError('coupled physics currently requires one flat ground at z=0')
    if not (planes[0].bounds[0]<=0 and planes[0].bounds[1]>=world.width_m
            and planes[0].bounds[2]<=0 and planes[0].bounds[3]>=world.height_m):
        raise ValueError('ground must cover the planning map')
    root=ET.fromstring(MODEL.read_bytes());body=root.find('worldbody')
    for i,box in enumerate(s for s in world.surfaces if isinstance(s,Box)):
        center=[(a+b)/2 for a,b in zip(box.low,box.high)]
        half=[(b-a)/2 for a,b in zip(box.low,box.high)]
        ET.SubElement(body,'geom',dict(name='obstacle_'+str(i),type='box',
                      pos=' '.join(map(str,center)),size=' '.join(map(str,half)),rgba='.7 .4 .3 1'))
    return ET.tostring(root,encoding='utf-8')


@dataclass(frozen=True)
class FlightBudget:
    body_radius_m: float=.35
    tracking_margin_m: float=.12
    stop_reserve_m: float=.30
    max_move_s: float=8.0
    free_ttl_s: float=12.0
    position_tolerance_m: float=.03
    speed_tolerance_mps: float=.03
    stable_s: float=.30
    height_tolerance_m: float=.10

    def __post_init__(self):
        if any(type(v) not in (int,float) or not isfinite(v) or v<=0 for v in asdict(self).values()):
            raise ValueError('flight budgets must be finite and positive')
        if self.body_radius_m+self.height_tolerance_m>=.5 or self.position_tolerance_m>self.tracking_margin_m:
            raise ValueError('budget does not fit fixed flight layer/tracking margin')
        if self.max_move_s>30 or self.stable_s>self.max_move_s or self.free_ttl_s<=self.max_move_s+1.4:
            raise ValueError('invalid motion duration or observation lifetime')


def spatial_cells(start,end,radius):
    return {(x,y,3) for x in range(floor(min(start[0],end[0])+.5-radius),floor(max(start[0],end[0])+.5+radius)+1)
            for y in range(floor(min(start[1],end[1])+.5-radius),floor(max(start[1],end[1])+.5+radius)+1)}


def physical_guard(grid,start,end,tick,now,stamps,budget):
    # 这是待进一步验证的工程预留，不借用旧版理想减速度公式声称物理停止保证。
    cells=spatial_cells(start,end,budget.body_radius_m+budget.tracking_margin_m+budget.stop_reserve_m)
    unsafe=sum(grid.state(c,tick)!='free' for c in cells)
    stale=sum(c in grid.free_seen and now+budget.max_move_s+.5-stamps[grid.free_seen[c]]>budget.free_ttl_s for c in cells)
    return dict(allowed=not unsafe and not stale,reason='BRAKE_MARGIN_HOLD' if unsafe else 'STALE_MAP_HOLD' if stale else 'APPROVED',
                unsafe_cells=unsafe,expiring_cells=stale,latest_stop_s=now+budget.max_move_s+.5,
                reserved_radius_m=budget.body_radius_m+budget.tracking_margin_m+budget.stop_reserve_m)


class PhysicalVehicle:
    def __init__(self,world,start,*,dll=DEFAULT_DLL,dropout_tick=None,ignore_move=None,budget=None):
        self.budget=budget or FlightBudget()
        self.physics=QuadrotorPhysics(dll,world_xml(world))
        self.physics.reset((start[0]+.5,start[1]+.5,3.5))
        self.world,self.dropout_tick,self.ignore_move=world,dropout_tick,ignore_move
        self.frames,self.commands,self.history,self.actions=[],[],[],[]
        self.history.append(self.physics.state())
        self.hold_target=(start[0]+.5,start[1]+.5,3.5)
        self.steps=0
        self.audit=dict(clearance_violations=0,boundary_violations=0,minimum_center_to_box_m=None,
                        scope='独立评价 0.002 s 状态间线段与 .35 m 球体；非连续时间严格证明，不参与选路')

    def close(self):self.physics.close()

    def state(self):return self.physics.state()

    def _step(self,target):
        before=self.state();motors,_=control(before,target)
        return self.apply_motors(motors)

    def apply_motors(self,motors):
        """控制与停桨共用同一物理推进、日志及独立评价入口。"""
        before=self.state()
        self.commands.append(motors);self.physics.step(motors);self.steps+=1
        after=self.state()
        if abs(after['time_s']-self.steps*self.physics.dt)>1e-6:
            raise ValueError('physical clock discontinuity')
        if self.steps%10==0:self.history.append(after)
        # 以下只积累评价，永远不用于规划器或控制器选择动作。
        for box in (s for s in self.world.surfaces if isinstance(s,Box)):
            distance=segment_box_clearance(before['position'],after['position'],box)
            old=self.audit['minimum_center_to_box_m']
            self.audit['minimum_center_to_box_m']=distance if old is None else min(old,distance)
            self.audit['clearance_violations']+=int(distance<=self.budget.body_radius_m)
        x,y,z=after['position'];radius=self.budget.body_radius_m
        self.audit['boundary_violations']+=int(min(x,y,self.world.width_m-x,self.world.height_m-y)<radius)
        return after

    def hold(self,seconds):
        for _ in range(round(seconds/self.physics.dt)):self._step(self.hold_target)

    def capture(self,intrinsics,tick,aim=None):
        state=self.state();xyz=tuple(state['position']);q=state['quaternion']
        base=Pose.look_at(xyz,(xyz[0],xyz[1],0)) if aim is None else aim(xyz)
        pose=Pose(xyz,rotate_vector(base.right,q),rotate_vector(base.down,q),rotate_vector(base.forward,q))
        frame,_=render(self.world,pose,intrinsics,tick=tick,seed=1701,
                       dropout=1.0 if tick==self.dropout_tick else 0)
        self.frames.append(dict(frame=asdict(frame),capture_state=state))
        return frame

    def scan(self,tick,aim):
        k=Intrinsics(40,30,25,25,19.5,14.5);frames=[]
        for i in range(8):
            frames.append(self.capture(k,tick,lambda xyz,i=i:scan_poses(xyz,aim)[i]))
            self.hold(.1)
        self.hold(.1)
        return frames

    def move(self,target_cell):
        b=self.budget;target=(target_cell[0]+.5,target_cell[1]+.5,3.5)
        before=self.state();stable=0;completed=False
        control_target=self.hold_target if len(self.actions)==self.ignore_move else target
        max_height_error=max_tracking_error=max_speed=0.0
        a=before['position'];direction=[target[j]-a[j] for j in range(2)]
        length_sq=sum(v*v for v in direction)
        breached=False
        for i in range(round(b.max_move_s/self.physics.dt)):
            state=self._step(control_target)
            distance=sqrt(sum((a-c)**2 for a,c in zip(state['position'],target)))
            speed=sqrt(sum(v*v for v in state['velocity']))
            max_height_error=max(max_height_error,abs(state['position'][2]-3.5))
            t=max(0,min(1,sum((state['position'][j]-a[j])*direction[j] for j in range(2))/max(length_sq,1e-12)))
            tracking=sqrt(sum((state['position'][j]-a[j]-t*direction[j])**2 for j in range(2)))
            max_tracking_error=max(max_tracking_error,tracking);max_speed=max(max_speed,speed)
            stable=stable+1 if distance<=b.position_tolerance_m and speed<=b.speed_tolerance_mps else 0
            if stable*self.physics.dt>=b.stable_s:
                completed=True;break
            if max_height_error>b.height_tolerance_m or tracking>b.tracking_margin_m:
                breached=True;break
        after=self.state()
        self.hold_target=target if completed else tuple(after['position'])
        # 未确认到达时不跳到格心；先以当前位置为停止目标，再结束任务。
        if not completed:self.hold(.5);after=self.state()
        receipt=dict(start=before,end=after,target=target,completed=completed,
                     duration_s=after['time_s']-before['time_s'],max_height_error_m=max_height_error,
                     max_tracking_error_m=max_tracking_error,max_speed_mps=max_speed,
                     position_error_m=sqrt(sum((a-c)**2 for a,c in zip(after['position'],target))),
                     speed_mps=sqrt(sum(v*v for v in after['velocity'])),stable_duration_s=stable*self.physics.dt,
                     stop_confirmed=sqrt(sum(v*v for v in after['velocity']))<=b.speed_tolerance_mps,
                     reason='ARRIVED_AND_SLOW' if completed else 'TRACKING_ENVELOPE_BREACH' if breached else 'CONTROLLER_TIMEOUT')
        self.actions.append(receipt)
        return receipt
