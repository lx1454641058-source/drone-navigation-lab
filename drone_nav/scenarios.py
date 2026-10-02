"""来源：本项目原创合成场景；不含真实校园地图或第三方数据。"""

from .mission import LandingSite, Scenario


def rectangle(x1: int, y1: int, x2: int, y2: int) -> set[tuple[int, int]]:
    return {(x, y) for x in range(x1, x2 + 1) for y in range(y1, y2 + 1)}


def scenarios() -> list[Scenario]:
    normal = LandingSite("宿舍区取餐点", (26, 10))
    building = rectangle(11, 7, 14, 13)
    return [
        Scenario("known_building", "01 · 绕开已知建筑", "已有地图包含建筑轮廓，规划时预留一格缓冲。",
                 sites=[normal], known_obstacles=building),
        Scenario("new_obstacle", "02 · 发现未知障碍后改道", "有限半径的模拟观测发现原地图未标注的障碍。障碍本身不移动。",
                 sites=[normal], hidden_obstacles=building),
        Scenario("blocked_route", "03 · 通道完全阻断", "障碍横贯地图，规划应报告无路可走。",
                 sites=[normal], hidden_obstacles=rectangle(14, 0, 15, 19)),
        Scenario("sensor_uncertain", "04 · 感知不可靠时停止", "第 5 个离散时刻注入低可信观测，停止继续运动。",
                 sites=[normal], degraded_tick=5),
        Scenario("water_diversion", "05 · 水面拒绝降落并改道", "第一处候选点为水面，到达后拒绝降落并前往备选点。",
                 sites=[LandingSite("原取餐点（水面）", (23, 10), surface="water"),
                        LandingSite("备用取餐点", (25, 4))]),
        Scenario("steep_landing", "06 · 陡坡拒绝降落", "目标点坡度过大且没有备选点，应报告无可用降落点。",
                 sites=[LandingSite("坡地取餐点", (26, 10), slope_deg=15)]),
        Scenario("rising_terrain", "07 · 绕开过高地形", "假设固定高度 8 米，地形高 6 米的区域离地间距不足 3 米，选择绕行。",
                 sites=[normal], terrain={cell: 6.0 for cell in building}),
    ]
