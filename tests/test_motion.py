"""来源：本项目原创。验证解析运动边界、过期/未知保护与完整观测链路。"""

from dataclasses import replace
import math
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from drone_nav.exploration_experiment import SimulatedDepthCamera, audit_motion, exploration_cases
from drone_nav.frontier_policy import FrontierPolicy
from drone_nav.motion import MotionConfig, corridor_cells, movement_guard, profile, simulate
from drone_nav.occupancy import VoxelMap
from drone_nav.timed_navigation import navigate
from drone_nav.verify_exploration import RecordedCamera
from drone_nav.verify_motion import verify


class KinematicsTests(unittest.TestCase):
    def test_short_edge_triangle(self):
        p = profile(1, MotionConfig())
        self.assertEqual((p.peak_mps, p.total_s, p.cruise_s), (1, 2, 0))
        self.assertEqual(p.at(0), (0, 0))
        self.assertEqual(p.at(1), (.5, 1))
        self.assertEqual(p.at(2), (1, 0))

    def test_long_edge_trapezoid(self):
        p = profile(10, MotionConfig())
        self.assertEqual((p.peak_mps, p.cruise_s, p.total_s), (2, 3, 7))
        self.assertEqual(p.at(7), (10, 0))

    def test_asymmetric_acceleration_and_braking(self):
        c = MotionConfig(acceleration_mps2=2, brake_mps2=.5)
        p = profile(3, c)
        self.assertAlmostEqual(p.at(p.total_s)[0], 3)
        self.assertAlmostEqual(p.at(p.total_s)[1], 0)

    def test_fault_continues_through_delay_then_brakes(self):
        c = MotionConfig()
        result = simulate(profile(1,c), c, .4)
        self.assertAlmostEqual(result['distance_m'], .24)
        self.assertAlmostEqual(result['reaction_distance_m'], .08)
        self.assertAlmostEqual(result['braking_distance_m'], .08)
        self.assertAlmostEqual(result['duration_s'], 1)
        self.assertFalse(result['completed_edge'])
        self.assertAlmostEqual(result['samples'][-1]['speed_mps'], 0)
        self.assertIn('REACTION_DELAY', {s['phase'] for s in result['samples']})

    def test_stop_envelope_bounds_fault_at_each_phase(self):
        for c in (MotionConfig(), MotionConfig(acceleration_mps2=3, brake_mps2=.4, reaction_s=.7)):
            for distance in (1, 10):
                p = profile(distance, c)
                for i in range(100):
                    r = simulate(p,c,p.total_s*i/100)
                    self.assertLessEqual(r['distance_m'], distance+p.peak_mps*c.reaction_s+1e-10)
                    self.assertLessEqual(r['duration_s'], p.total_s+c.reaction_s+1e-10)
                    self.assertTrue(all(b['distance_m']+1e-10>=a['distance_m']
                                        for a,b in zip(r['samples'],r['samples'][1:])))

    def test_fault_during_planned_braking_may_pass_waypoint(self):
        c = MotionConfig()
        r = simulate(profile(1,c), c, 1.2)
        self.assertAlmostEqual(r['distance_m'], 1.16)
        self.assertAlmostEqual(r['samples'][-1]['speed_mps'], 0)

    def test_zero_speed_fault(self):
        c = MotionConfig()
        r = simulate(profile(1,c),c,0)
        self.assertEqual(r['distance_m'], 0)

    def test_invalid_parameters(self):
        for kw in ({'max_speed_mps':0}, {'reaction_s':-1}, {'scan_s':math.nan},
                   {'body_radius_m':.5}, {'sample_dt_s':1}, {'brake_mps2':True}):
            with self.assertRaises(ValueError): MotionConfig(**kw)
        for value in (0, -1, math.inf, math.nan, True):
            with self.assertRaises(ValueError): profile(value,MotionConfig())

    def test_invalid_fault_times(self):
        c = MotionConfig()
        for value in (-1,2,math.nan,True):
            with self.assertRaises(ValueError): simulate(profile(1,c),c,value)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelMap(10,10)
        self.grid.free_seen = {(x,y,3):0 for x in range(10) for y in range(10)}
        self.c = MotionConfig()

    def guard(self, c=None):
        c = c or self.c
        return movement_guard(self.grid,(3,4),(4,4),0,.9,{0:0},profile(1,c),c)

    def test_free_fresh_corridor_allowed(self):
        self.assertTrue(self.guard()['allowed'])

    def test_unknown_after_edge_prevents_long_delay_departure(self):
        del self.grid.free_seen[(6,4,3)]
        result = self.guard(replace(self.c,reaction_s=3))
        self.assertFalse(result['allowed'])
        self.assertEqual(result['reason'],'BRAKE_MARGIN_HOLD')

    def test_fresh_now_but_expiring_before_stop(self):
        r = self.guard(replace(self.c,free_ttl_s=1))
        self.assertEqual(r['reason'],'STALE_MAP_HOLD')
        self.assertGreater(r['expiring_cells'],0)

    def test_occupied_wins(self):
        self.grid.occupied.add((4,4,3))
        self.assertFalse(self.guard()['allowed'])

    def test_negative_direction_corridor(self):
        cells = corridor_cells((3,4),(-1,0),1.35,.25)
        self.assertEqual(cells,{(x,4,3) for x in (1,2,3)})

    def test_no_diagonal_corridor(self):
        with self.assertRaises(ValueError): corridor_cells((3,4),(1,1),1,.25)

    def test_history_does_not_forget_when_free_expires(self):
        policy = FrontierPolicy()
        self.assertEqual(policy.update(self.grid,(3,4)),100)
        self.grid.free_seen.clear()
        self.assertEqual(policy.update(self.grid,(3,4)),0)
        self.assertEqual(len(policy.discovered),100)


class TimedNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = exploration_cases()
        cls.results = []
        for case in cls.cases:
            cls.results.append(navigate(SimulatedDepthCamera(case['world'],case['dropout_tick']),
                                        20,16,case['start'],case['goal'],previews=False))

    def test_reaches_target_and_independent_geometry_has_clearance(self):
        r = self.results[0]
        self.assertEqual(r['terminal_state'],'ARRIVED_WAYPOINT')
        self.assertEqual(r['trace'][-1]['position'],self.cases[0]['goal'])
        self.assertEqual(audit_motion(self.cases[0]['world'],r)['clearance_violations'],0)
        self.assertAlmostEqual(r['elapsed_s'],r['observations']*.9+len(r['moves'])*2)
        self.assertTrue(all(m['guard']['allowed'] for m in r['moves']))

    def test_wall_stops_exploring_before_budget(self):
        r = self.results[1]
        self.assertIn(r['terminal_state'],('EXPLORE_STALLED','NO_GAIN_FRONTIER'))
        self.assertLess(r['observations'],r['max_ticks'])

    def test_scan_failure_does_not_move_again(self):
        r = self.results[2]
        self.assertEqual(r['terminal_state'],'SENSOR_HOLD')
        self.assertEqual(len(r['moves']),3)

    def test_motion_fault_and_frame_replay(self):
        case = self.cases[0]
        camera = SimulatedDepthCamera(case['world'],record=True)
        r = navigate(camera,20,16,case['start'],case['goal'],failure_edge=0,previews=False)
        self.assertEqual(r['terminal_state'],'EMERGENCY_STOP')
        self.assertAlmostEqual(r['distance_m'],.24)
        self.assertAlmostEqual(r['trace'][-1]['position'][0],3.24)
        self.assertEqual(audit_motion(case['world'],r)['clearance_violations'],0)
        replay = navigate(RecordedCamera(camera.frames),20,16,case['start'],case['goal'],failure_edge=0,previews=False)
        self.assertEqual(r,replay)

    def test_long_delay_and_short_lifetime_hold(self):
        case = self.cases[0]
        for config,expected in ((MotionConfig(reaction_s=3),'BRAKE_MARGIN_HOLD'),
                                (MotionConfig(free_ttl_s=1),'STALE_MAP_HOLD')):
            r = navigate(SimulatedDepthCamera(case['world']),20,16,case['start'],case['goal'],config=config,previews=False)
            self.assertEqual(r['terminal_state'],expected)
            self.assertEqual(r['distance_m'],0)

    def test_invalid_mission_arguments(self):
        for kw in ({'max_ticks':0},{'failure_edge':-1},{'policy_mode':'other'}):
            with self.assertRaises(ValueError): navigate(None,20,16,(3,8),(16,8),**kw)

    def test_time_budget_does_not_duplicate_observation(self):
        case = self.cases[0]
        r = navigate(SimulatedDepthCamera(case['world']),20,16,case['start'],case['goal'],max_ticks=1,previews=False)
        self.assertEqual(r['terminal_state'],'TIMEOUT')
        self.assertEqual(len(r['trace']),r['observations'])


class SavedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case = exploration_cases()[0]
        camera = SimulatedDepthCamera(case['world'],record=True)
        cls.result = navigate(camera,20,16,case['start'],case['goal'],failure_edge=0,previews=False)
        cls.blob = json.dumps(camera.frames).encode('utf-8')

    def fixture(self, root):
        name = 'fault-observations.json'
        (root/name).write_bytes(self.blob)
        (root/'observations_manifest.json').write_text(json.dumps([
            {'path':name,'frames':8,'sha256':hashlib.sha256(self.blob).hexdigest()}]),encoding='utf-8')
        report = {'results':[dict(self.result,key='fault')]}
        (root/'motion_report.json').write_text(json.dumps(report),encoding='utf-8')
        return report

    def test_saved_frames_reconstruct_identical_mission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            self.assertTrue(verify(root)['verified'])

    def test_modified_raw_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (root/'fault-observations.json').write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError,'digest'): verify(root)

    def test_modified_motion_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = self.fixture(root)
            report['results'][0]['distance_m'] = 0
            (root/'motion_report.json').write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'differs'): verify(root)


if __name__ == '__main__': unittest.main()
