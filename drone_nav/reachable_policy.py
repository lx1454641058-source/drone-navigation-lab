"""来源：本项目原创。以整段停止条件建立有向可行图，在可达区域内探索。"""

from collections import deque

from .frontier_policy import FrontierPolicy


class ReachableFrontierPolicy(FrontierPolicy):
    """只复用观测历史和观察收益；边可否通行由传入的运动检查决定。"""

    def __init__(self):
        super().__init__()
        self.diagnostics = {}

    def route(self, grid, start, goal, tick, *, edge_check, can_stay):
        self.diagnostics = {'checked_edges':0,'rejected_space':0,'rejected_age':0,
                            'reachable_cells':0,'frontier_candidates':0}
        if not can_stay(start):
            return [],'NO_OBSERVED_START',0
        previous, distance = {start:None},{start:0}
        queue = deque([start])
        # 一次广度优先遍历得到所有可达点，避免逐个候选重复搜索。
        while queue:
            current = queue.popleft()
            x,y = current
            for nxt in ((x+1,y),(x,y+1),(x-1,y),(x,y-1)):
                if not (0 <= nxt[0] < grid.width and 0 <= nxt[1] < grid.height) or nxt in previous:
                    continue
                check = edge_check(current,nxt)
                self.diagnostics['checked_edges'] += 1
                if not check['allowed']:
                    key = 'rejected_age' if check['reason']=='STALE_MAP_HOLD' else 'rejected_space'
                    self.diagnostics[key] += 1
                    continue
                previous[nxt],distance[nxt] = current,distance[current]+1
                queue.append(nxt)
        self.diagnostics['reachable_cells'] = len(previous)

        def path_to(cell):
            path = [cell]
            while previous[cell] is not None:
                cell = previous[cell]
                path.append(cell)
            return list(reversed(path))

        if goal in previous:
            return path_to(goal),'GOAL_PATH',0
        if self.pending not in (None,start,goal) and self.pending in previous:
            gain = self.gain(grid,self.pending)
            if gain:
                return path_to(self.pending),'FRONTIER_PATH',gain
        candidates = []
        for cell in previous:
            if cell == start or cell in self.scanned:
                continue
            gain = self.gain(grid,cell)
            if gain:
                # 保留历史策略的候选评分，便于对照可行图与停止条件的影响。
                score = abs(cell[0]-goal[0])+abs(cell[1]-goal[1])+.6*distance[cell]-.12*min(gain,30)
                candidates.append((score,cell,gain))
        self.diagnostics['frontier_candidates'] = len(candidates)
        if not candidates:
            return [],'NO_FEASIBLE_FRONTIER',0
        _,self.pending,gain = min(candidates)
        return path_to(self.pending),'FRONTIER_PATH',gain
