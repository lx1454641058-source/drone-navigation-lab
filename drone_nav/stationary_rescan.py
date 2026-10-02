"""来源：本项目原创。历史帧账本与只转相机的运动前复查，不读取世界真值。"""
from copy import deepcopy
from math import sqrt

from .detection_bridge import number,identifier
from .motion import movement_guard,corridor_cells,profile,simulate
from .occupancy import VoxelMap
from .pinhole import Pose
from .sampling_motion import SamplingView,sampled_movement_guard


class SamplingHistory:
    """当前地图保留占用；逐帧独立来源保留旧观察的原年龄。"""
    def __init__(self,width=20,height=16,layers=8,clock_id='simulation-seconds'):
        if not identifier(clock_id):raise ValueError('clock identity required')
        self.grid=VoxelMap(width,height,layers,ttl_ticks=100)
        self.clock_id=clock_id;self.stamps={};self._entries=[];self._ids=set();self._now=-1.;self._capture=-1.

    def _time(self,now):
        if not number(now) or now<0 or now<self._now:raise ValueError('consumer time must increase')

    def add(self,view,*,now_s):
        self._time(now_s)
        if not isinstance(view,SamplingView):raise ValueError('invalid view')
        view.validate();key=(view.camera_id,view.frame_id)
        if (view.clock_id!=self.clock_id or view.available_at_s>now_s or view.captured_at_s<=self._capture or
                view.frame.tick<=self.grid.last_tick or key in self._ids):
            raise ValueError('source, timing, ordering or frame identity mismatch')
        if len(self._entries)>=128:raise ValueError('history budget exceeded')
        # 原体素算法可能在射线处理中发现异常。两个临时地图都成功后才提交。
        stored=deepcopy(view)
        single=VoxelMap(self.grid.width,self.grid.height,self.grid.layers,ttl_ticks=100)
        single.integrate_moving([stored.frame],stored.frame.tick)
        updated=deepcopy(self.grid);info=updated.integrate_moving([stored.frame],stored.frame.tick)
        self.grid=updated;self.stamps[stored.frame.tick]=stored.captured_at_s
        self._entries.append((stored,single));self._ids.add(key)
        self._now=now_s;self._capture=stored.captured_at_s
        return info

    def guard(self,start,end,*,now_s,config,assumption=None,error_bound_m=0.):
        self._time(now_s);p=profile(1,config);tick=max(0,self.grid.last_tick)
        base=movement_guard(self.grid,start,end,tick,now_s,self.stamps,p,config,error_bound_m=error_bound_m)
        required=sorted(corridor_cells(start,(end[0]-start[0],end[1]-start[1]),base['required_distance_m'],
                                      config.body_radius_m+2*error_bound_m))
        cells={c:dict(voxel=list(c),supported=False,candidates=[]) for c in required}
        for view,single in self._entries:
            # 对保存的实际单帧地图调用原采样检查，避免把旧帧的来源 tick 改成新 tick。
            result=sampled_movement_guard(single,start,end,tick,now_s,self.stamps,p,config,views=[view],
                                          assumption=assumption,clock_id=self.clock_id,error_bound_m=error_bound_m)
            for row in result['sampling']:
                item=cells[tuple(row['voxel'])];check=row['views'][0]
                item['candidates'].append(dict(**check,captured_at_s=view.captured_at_s,
                                               valid_until_s=view.captured_at_s+config.free_ttl_s))
                item['supported']|=row['supported']
        allowed=base['allowed'] and all(v['supported'] for v in cells.values())
        self._now=now_s
        return dict(allowed=allowed,baseline=base,required_cells=[list(c) for c in required],
            sampling=list(cells.values()),reason=base['reason'] if not base['allowed'] else
            'SAMPLING_CONDITION_HOLD' if not allowed else 'CONDITIONAL_SIMULATION_APPROVAL',
            history_frames=len(self._entries),flight_authorized=False)


PHASES=((0.,0.),(.5,0.),(0.,.5),(.5,.5))


def next_phase(used,mode):
    if mode not in ('repeat','phase_diverse'):raise ValueError('unknown rescan policy')
    if mode=='repeat' or not used:return PHASES[0]
    remaining=[p for p in PHASES if p not in used]
    if not remaining:raise ValueError('all phases already used')
    return max(remaining,key=lambda p:min(sum((a-b)**2 for a,b in zip(p,q)) for q in used))


def phase_pose(start,end,k,phase):
    direction=(end[0]-start[0],end[1]-start[1])
    if direction not in ((1,0),(-1,0),(0,1),(0,-1)):raise ValueError('adjacent axis-aligned move required')
    xyz=(start[0]+.5,start[1]+.5,3.5)
    base=Pose.look_at(xyz,(xyz[0]+direction[0],xyz[1]+direction[1],xyz[2]))
    ray=(phase[0]/k.fx,phase[1]/k.fy,1.)
    forward=base.rotate(ray)
    return Pose.look_at(xyz,tuple(x+d for x,d in zip(xyz,forward)))


def verify_stationary(history,source,start,end,k,*,config,assumption=None,mode='phase_diverse',
                      now_s=1.,max_scans=4,settle_s=.1,processing_s=.05):
    if type(max_scans) is not int or not 1<=max_scans<=4:raise ValueError('invalid scan budget')
    if any(not number(v) or v<=0 for v in (settle_s,processing_s)):raise ValueError('invalid scan duration')
    if mode not in ('repeat','phase_diverse'):raise ValueError('unknown rescan policy')
    initial=history.guard(start,end,now_s=now_s,config=config,assumption=assumption)
    events=[];used=[];decision=initial;state='SCAN_BUDGET_HOLD';start_s=now_s
    if any(tuple(c) in history.grid.occupied for c in initial['required_cells']):
        return dict(state='KNOWN_OBSTACLE_HOLD',initial=initial,final=initial,events=[],motion=None,
                    elapsed_scan_s=0.,flight_authorized=False)
    for _ in range(max_scans):
        phase=next_phase(used,mode);used.append(phase);pose=phase_pose(start,end,k,phase)
        tick=history.grid.last_tick+1;capture=now_s+settle_s;available=capture+processing_s
        try:
            view=source.capture(pose,k,tick,capture,available)
            if (view.frame.intrinsics!=k or view.frame.pose!=pose or view.frame.tick!=tick or
                    abs(view.captured_at_s-capture)>1e-9 or abs(view.available_at_s-available)>1e-9):
                raise ValueError('captured frame differs from requested pose, calibration or timing')
            info=history.add(view,now_s=available);now_s=available
            decision=history.guard(start,end,now_s=now_s,config=config,assumption=assumption)
        except (ValueError,TypeError,OverflowError) as exc:
            now_s=available;state='SCAN_PROTOCOL_HOLD'
            events.append(dict(phase=list(phase),time_s=now_s,state=state,error=str(exc),decision=None))
            break
        positive=any(tuple(c) in history.grid.occupied for c in decision['required_cells'])
        state='OBSERVED_OBSTACLE_HOLD' if positive else 'SCAN_SENSOR_HOLD' if info['valid_depth_fraction']<.05 else 'SCANNING'
        events.append(dict(phase=list(phase),time_s=now_s,state=state,observation=info,decision=decision))
        if state!='SCANNING':break
    else:
        state='READY_FOR_SIMULATED_MOVE' if decision['allowed'] else 'SCAN_BUDGET_HOLD'
    motion=simulate(profile(1,config),config) if state=='READY_FOR_SIMULATED_MOVE' else None
    return dict(state=state,initial=initial,final=decision,events=events,motion=motion,
                elapsed_scan_s=now_s-start_s,flight_authorized=False,
                limitation='静态理想相机转台和运动学；相位复查仍非任意细障碍的完整覆盖证明。')
