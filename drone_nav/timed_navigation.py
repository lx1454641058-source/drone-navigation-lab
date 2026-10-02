"""来源：本项目原创。停驻扫描、观测过期检查和带制动过程的固定高度导航。"""

from collections import Counter
from dataclasses import asdict

from .exploration import choose_route, preview, scan_poses
from .frontier_policy import FrontierPolicy
from .localization import LocalizationBudget
from .motion import MotionConfig, corridor_cells, movement_guard, profile, simulate
from .occupancy import VoxelMap
from .pinhole import Intrinsics
from .reachable_policy import ReachableFrontierPolicy


def navigate(camera, width, height, start, goal, *, config=None, max_ticks=70,
             failure_edge=None, failure_at_s=.4, previews=True, policy_mode='history', localization=None):
    config = config or MotionConfig()
    for cell in (start, goal):
        if (not isinstance(cell, tuple) or len(cell) != 2 or any(type(v) is not int for v in cell)
                or not 1 <= cell[0] < width-1 or not 1 <= cell[1] < height-1):
            raise ValueError('start and goal must be interior integer cells')
    if type(max_ticks) is not int or not 1 <= max_ticks <= 10000:
        raise ValueError('step budget must be between 1 and 10000')
    if failure_edge is not None and (type(failure_edge) is not int or failure_edge < 0):
        raise ValueError('failure edge must be nonnegative')
    if policy_mode not in ('history', 'legacy', 'reachable'):
        raise ValueError('unsupported exploration policy')
    if localization is not None and (not isinstance(localization,LocalizationBudget) or policy_mode == 'legacy'):
        raise ValueError('localization budget requires a supported policy and LocalizationBudget')
    error_bound = localization.axis_bound_m if localization else 0.0
    planning_margin = localization.planning_margin(config.body_radius_m) if localization else 1
    # 体素模块仍接收批次编号；有效期在本层按秒检查，不再用步数代替秒。
    grid = VoxelMap(width, height, ttl_ticks=max_ticks+1)
    policy = ReachableFrontierPolicy() if policy_mode == 'reachable' else FrontierPolicy()
    stamps = {}
    k = Intrinsics(40, 30, 25, 25, 19.5, 14.5)
    position, aim, now = start, goal, 0.0
    visits = Counter({start:1})
    trace, moves = [], []
    for tick in range(max_ticks):
        xyz = (position[0]+.5, position[1]+.5, 3.5)
        scan_start = now
        frames, info, path, guard, motion = [], {}, [], None, None
        gain = discovered = 0
        if policy_mode == 'reachable':
            policy.diagnostics = {}
        scan_layer = None
        state, reason = 'OBSERVE', ''
        try:
            frames = [camera.capture(pose, k, tick) for pose in scan_poses(xyz, aim)]
            info = grid.integrate(frames, tick, xyz)
            # 八帧时间统一取扫描开始，保守计算数据年龄；扫描耗时是配置假设。
            stamps[tick] = scan_start
            now += config.scan_s+config.processing_s
            discovered = policy.update(grid, position)
            expired = [cell for cell, stamp in grid.free_seen.items()
                       if now-stamps[stamp] > config.free_ttl_s]
            for cell in expired:
                del grid.free_seen[cell]
            info['expired_voxels'] = len(expired)
            info['free_voxels'] = sum(grid.state(cell, tick) == 'free' for cell in grid.free_seen)
            scan_layer = grid.layer(3, tick)
            if info['valid_depth_fraction'] < .05:
                state, reason = 'SENSOR_HOLD', '停驻扫描的有效深度不足 5%，保持静止'
            else:
                if policy_mode == 'reachable':
                    p = profile(1.0,config)
                    def edge_check(a,b):
                        return movement_guard(grid,a,b,tick,now,stamps,p,config,error_bound_m=error_bound)
                    def can_stay(cell):
                        footprint = corridor_cells(cell,(1,0),0,config.body_radius_m+2*error_bound)
                        return all(grid.state(c,tick)=='free' for c in footprint)
                    path,route,gain = policy.route(grid,position,goal,tick,edge_check=edge_check,can_stay=can_stay)
                elif policy_mode == 'history':
                    path, route, gain = policy.route(grid, position, goal, tick, planning_margin)
                else:
                    path, route = choose_route(grid, position, goal, tick, visits, pending=aim)
                if not path:
                    state, reason = route, '当前策略没有可继续探索的有效路线；不表示已证明全局无路'
                elif position == goal:
                    state, reason = 'ARRIVED_WAYPOINT', '抵达固定高度航点并停止；尚未降落、交付'
                    if localization and not localization.arrival_confirmable():
                        state, reason = 'UNCERTAIN_ARRIVAL', '估计位置到达，但定位误差范围超出目标容差，不能确认实际到达'
                else:
                    nxt = path[1]
                    p = profile(1.0, config)
                    guard = movement_guard(grid, position, nxt, tick, now, stamps, p, config, error_bound_m=error_bound)
                    if not guard['allowed']:
                        state = guard['reason']
                        reason = '停止所需区域尚未完整观测' if state == 'BRAKE_MARGIN_HOLD' else '预计停止前，相关空闲观测将过期'
                    else:
                        fault = failure_at_s if len(moves) == failure_edge else None
                        motion = simulate(p, config, fault)
                        direction = (nxt[0]-position[0], nxt[1]-position[1])
                        end = nxt if motion['completed_edge'] else tuple(
                            position[i]+direction[i]*motion['distance_m'] for i in range(2))
                        moves.append({'tick':tick, 'from':position, 'to':end,
                                      'start_s':now, 'end_s':now+motion['duration_s'],
                                      'route_kind':route, 'guard':guard, 'motion':motion})
                        position, aim = end, path[-1]
                        visits[position] += 1
                        now += motion['duration_s']
                        state = 'MOVE' if motion['completed_edge'] else 'EMERGENCY_STOP'
                        reason = ('完成一段加速、减速并停止，准备重新扫描' if state == 'MOVE' else
                                  '注入途中故障信号，经过反应延迟和减速后停止')
        except (ValueError, TypeError, OverflowError) as exc:
            state, reason = 'SENSOR_HOLD', '相机或运动输入无效：'+str(exc)
        # 页面展示停止时刻的地图，其他方向的旧空闲证据也必须按秒过期。
        for cell in [c for c, stamp in grid.free_seen.items() if now-stamps[stamp] > config.free_ttl_s]:
            del grid.free_seen[cell]
        trace.append({'tick':tick, 'state':state, 'reason':reason, 'position':position,
                      'observed_from':xyz, 'scan_start_s':scan_start, 'time_s':now,
                      'path':path, 'layer':grid.layer(3, tick), 'observation':info,
                      'scan_layer':scan_layer,
                      'new_cells':discovered, 'predicted_gain':gain, 'guard':guard, 'motion':motion,
                      'vision':preview(frames[0]) if frames and previews and info else None})
        if policy_mode == 'reachable':
            trace[-1]['planning_diagnostics'] = dict(policy.diagnostics)
        if state != 'MOVE':
            break
    else:
        trace[-1].update(state='TIMEOUT', reason='完成本段后达到观察预算，仿真结束')
    result = {'start':start, 'goal':goal, 'width':width, 'height':height, 'altitude_m':3.5,
            'max_ticks':max_ticks, 'config':asdict(config), 'intrinsics':asdict(k),
            'policy_mode':policy_mode,
            'failure_edge':failure_edge, 'failure_at_s':failure_at_s, 'trace':trace, 'moves':moves,
            'terminal_state':trace[-1]['state'], 'distance_m':sum(m['motion']['distance_m'] for m in moves),
            'elapsed_s':now, 'observations':len(stamps),
            'limitations':'静态理想深度、精确位姿、固定高度、每段停驻扫描。时延为设定值，非硬件测量。途中故障为信号注入；未模拟真实飞控、风或动态障碍。'}
    if localization is not None:
        result['localization_budget'] = asdict(localization)
        result['planning_margin_cells'] = planning_margin
        result['limitations'] = '静态深度、水平平移定位误差、准确方向和高度。误差上界来自配置；不是在线定位算法或实际飞控。'
    if policy_mode == 'reachable':
        result['planning_constraint'] = '每条候选边检查完整停止区域及观测时效；不额外叠加固定整格缓冲。路线仅为当前观测下的意向，每次出发重新检查。'
        # 新策略不使用旧版整格扩张；仍保留完全相同的机体、误差和制动参数。
        result['planning_margin_cells'] = None
    return result
