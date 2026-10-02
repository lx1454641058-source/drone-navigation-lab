"""来源：本项目原创。正向表面证据缓存及影子路线，不能授权飞行。"""
from copy import deepcopy
from itertools import product
from math import floor

from .detection_bridge import project_detections,BridgeConfig,number,identifier
from .planning import astar


class SurfaceEvidenceChannel:
    def __init__(self,width,height,layers,*,clock_id,config=None):
        if any(type(v) is not int or v<1 for v in (width,height,layers)) or not identifier(clock_id):
            raise ValueError('invalid channel dimensions or clock')
        self.width,self.height,self.layers=width,height,layers
        self.clock_id=clock_id;self.config=config or BridgeConfig()
        self._latest={};self._cells={};self._now=-1.0

    def _time(self,now):
        if not number(now) or now<0 or now<self._now:raise ValueError('consumer time must be monotonic')

    def _voxels(self,point):
        if len(point)!=3 or not all(number(v) for v in point):raise ValueError('invalid projected point')
        axes=[]
        for value in point:
            n=round(value)
            axes.append((n-1,n) if abs(value-n)<=1e-8 else (floor(value),))
        return [v for v in product(*axes) if all(0<=x<n for x,n in zip(v,(self.width,self.height,self.layers)))]

    def ingest(self,packet,context,*,now_s):
        self._time(now_s)
        # 完整投影和采样先完成；损坏输入抛错时不留下半批缓存或时间更新。
        result=project_detections(packet,context,now_s=now_s,now_clock_id=self.clock_id,config=self.config)
        previous=self._latest.get(packet.camera_id)
        reason=None
        if result['status']=='CONTEXT_REJECTED':reason='CONTEXT_REJECTED'
        elif previous is not None and packet.frame_id==previous['frame_id']:reason='REPEATED_FRAME'
        elif previous is not None and packet.captured_at_s<=previous['captured_at_s']:reason='OUT_OF_ORDER_CAPTURE'
        if reason:
            self._now=now_s
            return dict(accepted=False,reason=reason,projection=result,updated_cells=0)
        staged={};outside=0
        for o in result['observations']:
            for sample in o['samples']:
                cells=self._voxels(sample['world_m'])
                if not cells:outside+=1
                for cell in cells:
                    staged.setdefault(cell,[]).append(dict(detection=o['detection']['id'],pixel=list(sample['pixel']),
                        reported_group=o['detection']['group'],association='unverified_box_surface',world_m=list(sample['world_m'])))
        updated=0
        for cell,refs in staged.items():
            old=self._cells.get(cell)
            # 不同时刻/相机的较旧到达记录不能缩短或刷新更晚的证据。
            if old is None or packet.captured_at_s>old['captured_at_s']:
                self._cells[cell]=dict(camera_id=packet.camera_id,frame_id=packet.frame_id,captured_at_s=packet.captured_at_s,
                    valid_until_s=result['valid_until_s'],detector_source=packet.source,sources=deepcopy(result['sources']),references=refs)
                updated+=1
        self._latest[packet.camera_id]=dict(frame_id=packet.frame_id,captured_at_s=packet.captured_at_s)
        self._now=now_s
        return dict(accepted=True,reason='SURFACES_RECORDED' if staged else 'NO_SURFACES_NO_CLEARING',
                    projection=result,updated_cells=updated,out_of_bounds_samples=outside)

    def snapshot(self,now_s):
        self._time(now_s);self._now=now_s
        return dict(clock_id=self.clock_id,now_s=now_s,cells=[dict(voxel=list(cell),
            state='surface_restriction' if now_s<=v['valid_until_s']+1e-9 else 'unknown_after_expiry',
            **deepcopy(v)) for cell,v in sorted(self._cells.items())],flight_authorized=False)


def shadow_route(grid,channel,*,layer,tick,now_s,start,goal):
    """基础几何地图来自独立调用者；检测样本只能增加阻塞。零栅格缓冲仅供对照。"""
    if (grid.width,grid.height,grid.layers)!=(channel.width,channel.height,channel.layers):raise ValueError('map shape mismatch')
    if type(layer) is not int or not 0<=layer<grid.layers or type(tick) is not int or tick<grid.last_tick:
        raise ValueError('invalid geometry layer or time')
    for cell in (start,goal):
        if len(cell)!=2 or any(type(v) is not int for v in cell):raise ValueError('invalid route endpoint')
    snapshot=channel.snapshot(now_s);base=grid.blocked(layer,tick)
    restrictions={(v['voxel'][0],v['voxel'][1]) for v in snapshot['cells'] if v['voxel'][2]==layer}
    base_route=astar(grid.width,grid.height,start,goal,base,margin=0)
    route=astar(grid.width,grid.height,start,goal,base|restrictions,margin=0)
    return dict(status='ROUTE_FOR_REVIEW' if route else 'HOLD',baseline_route=base_route,route=route,
                base_blocked=[list(c) for c in sorted(base)],restrictions=[list(c) for c in sorted(restrictions)],
                layer=layer,geometry_tick=tick,observation=snapshot,flight_authorized=False)
