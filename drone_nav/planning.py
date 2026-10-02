"""来源：本项目原创。四邻接栅格 A*，只用于离散规划基线。"""

from heapq import heappop, heappush

Cell = tuple[int, int]


def inflate(obstacles: set[Cell], margin: int) -> set[Cell]:
    """用正方形缓冲区扩大障碍物，为机体和定位误差预留空间。"""
    if type(margin) is not int or margin < 0:
        raise ValueError("margin must be a non-negative integer")
    return {
        (x + dx, y + dy)
        for x, y in obstacles
        for dx in range(-margin, margin + 1)
        for dy in range(-margin, margin + 1)
    }


def astar(width: int, height: int, start: Cell, goal: Cell,
          blocked: set[Cell], margin: int = 1) -> list[Cell]:
    """路径包括起终点；不可达返回空列表；地图外也保留边界缓冲。"""
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("grid dimensions must be positive integers")
    forbidden = inflate(blocked, margin)

    def allowed(cell: Cell) -> bool:
        x, y = cell
        return (margin <= x < width - margin
                and margin <= y < height - margin and cell not in forbidden)

    if not allowed(start) or not allowed(goal):
        return []

    def estimate(cell: Cell) -> int:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    queue = [(estimate(start), 0, start)]
    costs = {start: 0}
    previous: dict[Cell, Cell] = {}
    while queue:
        _, cost, current = heappop(queue)
        if costs.get(current) != cost:
            continue
        if current == goal:
            path = [current]
            while current in previous:
                current = previous[current]
                path.append(current)
            return list(reversed(path))
        x, y = current
        for nxt in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            new_cost = cost + 1
            if allowed(nxt) and new_cost < costs.get(nxt, float("inf")):
                costs[nxt] = new_cost
                previous[nxt] = current
                heappush(queue, (new_cost + estimate(nxt), new_cost, nxt))
    return []
