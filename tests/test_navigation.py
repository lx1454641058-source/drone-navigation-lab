"""来源：本项目原创。检查路线性质、异常观测和交付决策边界。"""

import math
import unittest
from collections import deque
from dataclasses import replace
from random import Random

from drone_nav.mission import LandingSite, Observation, Scenario, landing_rejection, run_mission
from drone_nav.planning import astar, inflate
from drone_nav.scenarios import scenarios


class PlanningTests(unittest.TestCase):
    def test_shortest_empty_grid(self):
        path = astar(10, 10, (1, 1), (8, 8), set())
        self.assertEqual(len(path) - 1, 14)
        self.assertEqual(path[0], (1, 1))
        self.assertEqual(path[-1], (8, 8))

    def test_disconnected(self):
        self.assertEqual(astar(10, 10, (2, 5), (7, 5), {(5, y) for y in range(10)}), [])

    def test_start_equals_goal(self):
        self.assertEqual(astar(10, 10, (4, 4), (4, 4), set()), [(4, 4)])

    def test_blocked_start_or_goal(self):
        for obstacle in ((2, 2), (7, 7)):
            self.assertEqual(astar(10, 10, (2, 2), (7, 7), {obstacle}), [])

    def test_boundary_margin(self):
        self.assertEqual(astar(10, 10, (0, 4), (7, 7), set()), [])

    def test_narrow_gap_is_not_flyable(self):
        wall = {(5, y) for y in range(10) if y != 5}
        self.assertTrue(astar(10, 10, (2, 5), (7, 5), wall, margin=0))
        self.assertFalse(astar(10, 10, (2, 5), (7, 5), wall, margin=1))

    def test_invalid_margin(self):
        with self.assertRaises(ValueError):
            inflate(set(), -1)

    def test_random_maps_match_independent_breadth_first_search(self):
        """随机地图用独立广度优先搜索核验可达性和最短距离，而非硬编码路径。"""
        rng = Random(20260916)
        for _ in range(30):
            blocked = {(x, y) for x in range(8) for y in range(8) if rng.random() < 0.2}
            blocked.difference_update({(0, 0), (7, 7)})
            pending = deque([((0, 0), 0)])
            visited = {(0, 0)}
            distance = None
            while pending:
                (x, y), cost = pending.popleft()
                if (x, y) == (7, 7):
                    distance = cost
                    break
                for neighbor in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    if (0 <= neighbor[0] < 8 and 0 <= neighbor[1] < 8
                            and neighbor not in blocked and neighbor not in visited):
                        visited.add(neighbor)
                        pending.append((neighbor, cost+1))
            path = astar(8, 8, (0, 0), (7, 7), blocked, margin=0)
            self.assertEqual(len(path)-1 if path else None, distance)


class MissionTests(unittest.TestCase):
    def test_scenario_terminal_states(self):
        expected = ["READY_TO_LAND", "READY_TO_LAND", "NO_PATH", "SENSOR_HOLD",
                    "READY_TO_LAND", "NO_SAFE_SITE", "READY_TO_LAND"]
        for scenario, state in zip(scenarios(), expected):
            with self.subTest(scenario=scenario.key):
                self.assertEqual(run_mission(scenario)["metrics"]["terminal_state"], state)

    def test_all_demo_motion_keeps_clearance_from_world_truth(self):
        for scenario in scenarios():
            truth = scenario.known_obstacles | scenario.hidden_obstacles | scenario.terrain_blocked
            trace = run_mission(scenario)["trace"]
            previous = scenario.start
            for frame in trace:
                x, y = frame["position"]
                self.assertGreaterEqual(x, scenario.margin_cells)
                self.assertGreaterEqual(y, scenario.margin_cells)
                self.assertLess(x, scenario.width - scenario.margin_cells)
                self.assertLess(y, scenario.height - scenario.margin_cells)
                # 直接计算切比雪夫距离，避免复用被测 inflate 隐藏同一个错误。
                for ox, oy in truth:
                    self.assertGreater(max(abs(x-ox), abs(y-oy)), scenario.margin_cells)
                self.assertLessEqual(abs(x-previous[0]) + abs(y-previous[1]), 1)
                previous = (x, y)

    def test_unknown_obstacle_requires_replan(self):
        result = run_mission(scenarios()[1])
        self.assertGreaterEqual(result["metrics"]["planning_calls"], 2)
        self.assertFalse(result["trace"][0]["detected"])
        self.assertTrue(result["trace"][-1]["detected"])

    def test_low_confidence_does_not_move(self):
        result = run_mission(scenarios()[3])
        self.assertEqual(result["trace"][-1]["position"], result["trace"][-2]["position"])
        self.assertEqual(result["metrics"]["grid_moves"], 5)

    def test_stale_future_invalid_and_nan_observations(self):
        class BadSensor:
            def __init__(self, observation):
                self.observation = observation

            def observe(self, scenario, position, tick):
                return self.observation

        for observation in [Observation(-1, frozenset(), 1), Observation(1, frozenset(), 1),
                            Observation(0, frozenset(), math.nan),
                            Observation(0, frozenset(), 1, valid=False),
                            Observation(0, frozenset(), 1.1)]:
            result = run_mission(scenarios()[0], sensor=BadSensor(observation))
            self.assertEqual(result["metrics"]["terminal_state"], "SENSOR_HOLD")
            self.assertEqual(result["metrics"]["grid_moves"], 0)

    def test_landing_fallback(self):
        result = run_mission(scenarios()[4])
        self.assertEqual(result["metrics"]["rejected_sites"], ["原取餐点（水面）"])
        self.assertEqual(result["trace"][-1]["position"], (25, 4))

    def test_unreachable_primary_uses_alternative(self):
        scenario = Scenario("unreachable", "test", "test", sites=[LandingSite("A", (15, 10)),
                           LandingSite("B", (5, 5))], known_obstacles={(15, 10)})
        result = run_mission(scenario)
        self.assertEqual(result["metrics"]["terminal_state"], "READY_TO_LAND")
        self.assertEqual(result["trace"][-1]["site"], "B")

    def test_step_budget(self):
        result = run_mission(scenarios()[0], max_steps=1)
        self.assertEqual(result["metrics"]["terminal_state"], "TIMEOUT")
        self.assertEqual(result["metrics"]["grid_moves"], 1)

    def test_missing_destination(self):
        with self.assertRaises(ValueError):
            run_mission(Scenario("empty", "empty", "empty"))

    def test_sensor_must_see_next_step_margin(self):
        with self.assertRaises(ValueError):
            run_mission(replace(scenarios()[0], sensor_radius_cells=1))


class LandingTests(unittest.TestCase):
    def test_acceptable_boundary(self):
        self.assertIsNone(landing_rejection(LandingSite("pad", (1, 1), slope_deg=5,
                                                        clear_radius_m=2, confidence=0.8)))

    def test_rejects_unsafe_missing_and_corrupt_data(self):
        site = LandingSite("pad", (1, 1))
        for changes in ({"slope_deg": 5.01}, {"clear_radius_m": 1.99}, {"occupied": True},
                        {"surface": "unknown"}, {"surface": "water"}, {"confidence": 0.79},
                        {"confidence": math.nan}, {"slope_deg": -1}, {"clear_radius_m": math.inf},
                        {"occupied": None}, {"confidence": True}):
            with self.subTest(changes=changes):
                self.assertIsNotNone(landing_rejection(replace(site, **changes)))


if __name__ == "__main__":
    unittest.main()
