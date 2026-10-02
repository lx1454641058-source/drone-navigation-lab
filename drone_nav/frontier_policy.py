"""来源：本项目原创。利用历史发现和预期新观测减少重复探索的启发式策略。"""

from .occupancy import ray_voxels
from .planning import astar, inflate


class FrontierPolicy:
    """发现历史仅用于选择观察位置；能否移动仍由当前有效空闲地图决定。"""

    def __init__(self):
        self.discovered = set()
        self.scanned = set()
        self.pending = None
        self.no_gain_scans = 0

    def update(self, grid, position):
        seen = {(x, y) for x, y, z in grid.free_seen if z == 3}
        seen.update((x, y) for x, y, z in grid.occupied if z == 3)
        new_cells = len(seen-self.discovered)
        self.discovered.update(seen)
        self.scanned.add(position)
        self.no_gain_scans = self.no_gain_scans+1 if new_cells == 0 else 0
        return new_cells

    def gain(self, grid, candidate):
        x, y = candidate
        gain = 0
        for yy in range(max(0, y-4), min(grid.height, y+5)):
            for xx in range(max(0, x-4), min(grid.width, x+5)):
                if (xx, yy) in self.discovered:
                    continue
                # 用已知占用阻断预测视线，不用世界真值预测可见性。
                ray = ray_voxels((x+.5, y+.5, 3.5), (xx+.5, yy+.5, 3.5))
                if not any(cell in grid.occupied for cell in ray):
                    gain += 1
        return gain

    def route(self, grid, start, goal, tick, margin=1):
        blocked = grid.blocked(3, tick)
        direct = astar(grid.width, grid.height, start, goal, blocked, margin)
        if direct:
            return direct, 'GOAL_PATH', 0
        if self.pending not in (None, start, goal) and self.gain(grid, self.pending) > 0:
            path = astar(grid.width, grid.height, start, self.pending, blocked, margin)
            if path:
                return path, 'FRONTIER_PATH', self.gain(grid, self.pending)
        # 无新观测只触发有限探索结束，不宣称全局不存在路线。
        if self.no_gain_scans >= 3:
            return [], 'EXPLORE_STALLED', 0
        forbidden = inflate(blocked, margin)
        choices = []
        for y in range(1, grid.height-1):
            for x in range(1, grid.width-1):
                cell = (x, y)
                if cell in forbidden or cell in self.scanned:
                    continue
                gain = self.gain(grid, cell)
                if gain == 0:
                    continue
                path = astar(grid.width, grid.height, start, cell, blocked, margin)
                if path:
                    score = abs(x-goal[0])+abs(y-goal[1]) + .6*(len(path)-1) - .12*min(gain, 30)
                    choices.append((score, cell, path, gain))
        if not choices:
            return [], 'NO_GAIN_FRONTIER', 0
        _, self.pending, path, gain = min(choices)
        return path, 'FRONTIER_PATH', gain
