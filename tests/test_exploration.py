"""来源：本项目原创。观测融合、未知/过期处理与透视导航的回归测试。"""

from collections import Counter
from dataclasses import replace
import math
import unittest

from drone_nav.exploration import choose_route, explore
from drone_nav.exploration_experiment import SimulatedDepthCamera, audit_motion, exploration_cases, segment_box_distance
from drone_nav.occupancy import VoxelMap, ray_voxels
from drone_nav.pinhole import Intrinsics, PerspectiveFrame, Pose
from drone_nav.raycast import Box


class TraversalTests(unittest.TestCase):
    def test_axis_rays_stop_at_endpoint(self):
        self.assertEqual(ray_voxels((.5, .5, .5), (3.2, .5, .5)),
                         [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
        self.assertEqual(ray_voxels((3.2, .5, .5), (.5, .5, .5)),
                         [(3, 0, 0), (2, 0, 0), (1, 0, 0), (0, 0, 0)])

    def test_diagonal_ties_do_not_free_corner_neighbors(self):
        self.assertEqual(ray_voxels((.5, .5, .5), (2.5, 2.5, 2.5)),
                         [(0, 0, 0), (1, 1, 1), (2, 2, 2)])

    def test_negative_coordinates_and_zero_length(self):
        self.assertEqual(ray_voxels((-.5, .5, .5), (-2.5, .5, .5)),
                         [(-1, 0, 0), (-2, 0, 0), (-3, 0, 0)])
        self.assertEqual(ray_voxels((.5, .5, .5), (.5, .5, .5)), [(0, 0, 0)])

    def test_invalid_or_unbounded_ray_rejected(self):
        for endpoint in ((math.nan, 0, 0), (20000, 0, 0)):
            with self.assertRaises(ValueError):
                ray_voxels((0, 0, 0), endpoint)


class OccupancyTests(unittest.TestCase):
    def frame(self, depth=4.0, tick=0):
        # 两条几乎平行的射线，独立像素为穿越格提供至少两票。
        k = Intrinsics(2, 1, 1000, 1000, .5, 0)
        pose = Pose.look_at((1.5, 3.5, 3.5), (8, 3.5, 3.5))
        return PerspectiveFrame(k, pose, ((100, 100, 100),)*2, (depth, depth), tick)

    def test_free_before_hit_occupied_at_hit_unknown_behind(self):
        grid, frame = VoxelMap(10, 8), self.frame()
        grid.integrate([frame], 0, frame.pose.position)
        self.assertEqual(grid.state((3, 3, 3), 0), 'free')
        self.assertEqual(grid.state((5, 3, 3), 0), 'occupied')
        self.assertEqual(grid.state((6, 3, 3), 0), 'unknown')

    def test_missing_depth_never_clears_space(self):
        grid, frame = VoxelMap(10, 8), self.frame(None)
        info = grid.integrate([frame], 0, frame.pose.position)
        self.assertEqual(info['rays'], 0)
        self.assertEqual(grid.free_seen, {})
        self.assertEqual(grid.occupied, set())

    def test_hit_wins_over_free_rays_and_later_observations(self):
        grid, near, far = VoxelMap(10, 8), self.frame(2), self.frame(5)
        grid.integrate([near, far], 0, near.pose.position)
        self.assertEqual(grid.state((3, 3, 3), 0), 'occupied')
        grid.integrate([replace(far, tick=1)], 1, far.pose.position)
        self.assertEqual(grid.state((3, 3, 3), 1), 'occupied')

    def test_free_expires_but_static_hits_remain(self):
        grid, frame = VoxelMap(10, 8, ttl_ticks=2), self.frame()
        grid.integrate([frame], 0, frame.pose.position)
        self.assertEqual(grid.state((3, 3, 3), 2), 'free')
        self.assertEqual(grid.state((3, 3, 3), 3), 'unknown')
        self.assertEqual(grid.state((5, 3, 3), 3), 'occupied')

    def test_bad_batch_is_atomic(self):
        for bad in (replace(self.frame(), tick=1), replace(self.frame(), depth_z_m=(math.nan, 1)),
                    self.frame(31)):
            grid, frame = VoxelMap(10, 8), self.frame()
            with self.assertRaises(ValueError):
                grid.integrate([frame, bad], 0, frame.pose.position)
            self.assertEqual(grid.free_seen, {})
            self.assertEqual(grid.occupied, set())
            self.assertEqual(grid.last_tick, -1)

    def test_duplicate_tick_and_wrong_pose_rejected(self):
        grid, frame = VoxelMap(10, 8), self.frame()
        grid.integrate([frame], 0, frame.pose.position)
        with self.assertRaises(ValueError):
            grid.integrate([frame], 0, frame.pose.position)
        with self.assertRaises(ValueError):
            grid.integrate([replace(frame, tick=1)], 1, (2.5, 3.5, 3.5))

    def test_unknown_and_expired_cells_cannot_be_planned_through(self):
        grid = VoxelMap(10, 8, ttl_ticks=2)
        for x in range(6):
            for y in range(8):
                grid.free_seen[(x, y, 3)] = 0
        path, kind = choose_route(grid, (2, 3), (8, 3), 0, Counter())
        self.assertTrue(path)
        self.assertEqual(kind, 'FRONTIER_PATH')
        self.assertTrue(all(x < 5 for x, y in path))
        self.assertEqual(choose_route(grid, (2, 3), (8, 3), 3, Counter())[0], [])

    def test_pending_frontier_is_retained_if_still_reachable(self):
        grid = VoxelMap(10, 8)
        grid.free_seen = {(x,y,3):0 for x in range(6) for y in range(8)}
        path, _ = choose_route(grid, (2, 3), (8, 3), 0, Counter(), pending=(3, 5))
        self.assertEqual(path[-1], (3, 5))


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = exploration_cases()[0]
        cls.result = explore(SimulatedDepthCamera(cls.case['world']),20,16,
                             cls.case['start'],cls.case['goal'],previews=False)

    def test_reaches_goal_after_exploration(self):
        self.assertEqual(self.result['terminal_state'], 'ARRIVED_WAYPOINT')
        self.assertEqual(self.result['trace'][-1]['position'], self.case['goal'])
        self.assertTrue(any(m['route_kind'] == 'FRONTIER_PATH' for m in self.result['moves']))
        self.assertGreater(self.result['distance_m'], 13)

    def test_every_move_has_fresh_free_margin(self):
        frames = {f['tick']:f for f in self.result['trace']}
        for move in self.result['moves']:
            x,y = move['to']
            self.assertEqual(abs(move['from'][0]-x)+abs(move['from'][1]-y),1)
            self.assertTrue(all(frames[move['tick']]['layer'][y+dy][x+dx] == 'free'
                                for dx in (-1,0,1) for dy in (-1,0,1)))

    def test_independent_geometry_audit(self):
        audit = audit_motion(self.case['world'], self.result)
        self.assertEqual(audit['clearance_violations'], 0)
        self.assertGreater(audit['minimum_center_to_box_m'], .25)
        box = Box('box', 3, (2,2,0), (4,4,6))
        self.assertEqual(segment_box_distance((1,3,3.5),(5,3,3.5),box),0)
        self.assertEqual(segment_box_distance((1,1,3.5),(5,1,3.5),box),1)

    def test_depth_failure_stops_despite_map_history(self):
        result = explore(SimulatedDepthCamera(self.case['world'], 1),20,16,
                         self.case['start'],self.case['goal'],previews=False)
        self.assertEqual(result['terminal_state'],'SENSOR_HOLD')
        self.assertEqual(result['distance_m'],1)
        self.assertEqual(result['trace'][-1]['position'],result['trace'][-2]['position'])

    def test_stale_camera_cannot_move(self):
        delegate = SimulatedDepthCamera(self.case['world'])
        class BadCamera:
            def capture(self, pose, k, tick): return replace(delegate.capture(pose,k,tick), tick=tick+1)
        result = explore(BadCamera(),20,16,self.case['start'],self.case['goal'],previews=False)
        self.assertEqual(result['terminal_state'],'SENSOR_HOLD')
        self.assertEqual(result['moves'],[])

    def test_zero_budget_rejected(self):
        with self.assertRaises(ValueError):
            explore(SimulatedDepthCamera(self.case['world']),20,16,self.case['start'],self.case['goal'],max_ticks=0)

    def test_blocked_case_exhausts_budget_without_crossing_wall(self):
        case = exploration_cases()[1]
        result = explore(SimulatedDepthCamera(case['world']),20,16,case['start'],case['goal'],
                         max_ticks=4,previews=False)
        self.assertEqual(result['terminal_state'],'TIMEOUT')
        self.assertEqual(result['distance_m'],4)
        self.assertTrue(all(f['position'][0] < 8 for f in result['trace']))
        self.assertEqual(audit_motion(case['world'],result)['clearance_violations'],0)


if __name__ == '__main__':
    unittest.main()
