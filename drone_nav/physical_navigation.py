"""来源：本项目原创。观测—选路—物理执行—回执—再观测，规划器不接收世界真值。"""

from dataclasses import asdict
from math import sqrt

from .exploration import preview
from .occupancy import VoxelMap
from .perspective_landing import inspect_landing,landing_preview
from .physical_vehicle import FlightBudget,physical_guard,spatial_cells
from .pinhole import Intrinsics
from .reachable_policy import ReachableFrontierPolicy


def navigate_physical(vehicle,model,start,goal,*,max_ticks=40,previews=True):
    if type(max_ticks) is not int or not 1<=max_ticks<=100:
        raise ValueError('invalid physical observation budget')
    for c in (start,goal):
        if len(c)!=2 or any(type(v) is not int for v in c) or not 1<=c[0]<19 or not 1<=c[1]<15:
            raise ValueError('goal/start must be interior map cells')
    grid=VoxelMap(20,16,ttl_ticks=max_ticks+1);policy=ReachableFrontierPolicy()
    budget=vehicle.budget;position=start;aim=goal;trace=[];stamps={}
    landing=images=None;terminal='OBSERVATION_LIMIT'
    for tick in range(max_ticks):
        scan_start=vehicle.state()['time_s']
        try:
            frames=vehicle.scan(tick,aim)
            now=vehicle.state()['time_s'];stamps[tick]=scan_start
            info=grid.integrate_moving(frames,tick)
        except (ValueError,TypeError,OverflowError) as exc:
            terminal='SENSOR_PROTOCOL_HOLD'
            trace.append(dict(tick=tick,state=terminal,reason='本批观测协议异常：'+str(exc),position=position,
                              actual=vehicle.state(),scan_start_s=scan_start,scan_end_s=vehicle.state()['time_s'],
                              layer=grid.layer(3,tick),path=[],receipt=None,observation={},planning_diagnostics={},vision=None))
            break
        for c in [c for c,t in grid.free_seen.items() if now-stamps[t]>budget.free_ttl_s]:del grid.free_seen[c]
        policy.update(grid,position)
        path=[];receipt=None
        if info['valid_depth_fraction']<.05:
            terminal='SENSOR_HOLD';reason='本批深度无效，停止任务，不继续发出航点'
        else:
            def edge(a,b):return physical_guard(grid,a,b,tick,now,stamps,budget)
            def stay(c):return all(grid.state(v,tick)=='free' for v in spatial_cells(c,c,budget.body_radius_m+budget.tracking_margin_m))
            path,state,_=policy.route(grid,position,goal,tick,edge_check=edge,can_stay=stay)
            if not path:
                terminal=state;reason='近期观测不能支持当前策略的下一段路线'
            elif position==goal:
                actual=vehicle.state();target=(goal[0]+.5,goal[1]+.5,3.5)
                error=sqrt(sum((a-b)**2 for a,b in zip(actual['position'],target)))
                speed=sqrt(sum(v*v for v in actual['velocity']))
                if error>budget.position_tolerance_m or speed>budget.speed_tolerance_mps:
                    terminal='ARRIVAL_UNCONFIRMED';reason='目标格已到，但物理位置或速度未满足确认条件'
                else:
                    k=Intrinsics(96,72,45,45,47.5,35.5)
                    try:
                        frame=vehicle.capture(k,tick)
                        if frame.intrinsics!=k:raise ValueError('landing camera calibration mismatch')
                        landing,labels=inspect_landing(frame,model,(goal[0]+.5,goal[1]+.5,0),
                            expected_tick=tick,expected_position=actual['position'])
                        images=landing_preview(frame,landing,labels) if previews else None
                        terminal='READY_TO_LAND' if landing['accepted'] else 'LANDING_REJECTED';reason=landing['reason']
                    except (ValueError,TypeError,OverflowError) as exc:
                        terminal='LANDING_SENSOR_HOLD';reason='末端观测异常：'+str(exc)
            else:
                receipt=vehicle.move(path[1])
                if receipt['completed']:
                    position=path[1];aim=path[-1];terminal='MOVE';reason='物理位置和速度已持续满足阈值，确认本段完成'
                else:terminal=receipt['reason'];reason='控制器未完成本段，保留真实位置并结束任务'
        trace.append(dict(tick=tick,state=terminal,reason=reason,position=position,actual=vehicle.state(),
                          scan_start_s=scan_start,scan_end_s=now,layer=grid.layer(3,tick),path=path,receipt=receipt,
                          observation=info,planning_diagnostics=dict(policy.diagnostics),
                          vision=preview(frames[0]) if previews else None))
        if terminal!='MOVE':break
    else:
        terminal='OBSERVATION_LIMIT';trace[-1].update(state=terminal,reason='到达开发观察预算，结束本次任务')
    return dict(start=start,goal=goal,budget=asdict(budget),max_ticks=max_ticks,trace=trace,
                terminal_state=terminal,landing=landing,landing_images=images,model_digest=model.digest,
                elapsed_s=vehicle.state()['time_s'],actual_final=vehicle.state(),
                limitations='原创理想 RGB-D、精确仿真位姿、扫描转台、固定巡航高度；未下降交付，未实现自然图像识别。停止预留是开发参数，不是安全保证。')
