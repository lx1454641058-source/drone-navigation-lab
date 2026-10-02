"""来源：本项目原创。物理空间/期限、来源一致性及实际反馈的回归测试。"""
from copy import deepcopy
import unittest
from unittest.mock import patch

from drone_nav.native_physics import DEFAULT_DLL
from drone_nav.physical_rescan import PhysicalSamplingHistory, rescan_and_move
from drone_nav.physical_vehicle import FlightBudget, PhysicalVehicle, spatial_cells
from drone_nav.pinhole import Intrinsics, Pose
from drone_nav.sampling_motion import SamplingAssumption, SamplingView
from drone_nav.raycast import render
from tools.physical_rescan_experiment import specs, world_for, simulate_case


class PhysicalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = specs()[1]
        cls.world = world_for(cls.spec)
        cls.k = Intrinsics(160, 120, 50, 50, 79.5, 59.5)
        frame, _ = render(cls.world, Pose.look_at((1.5, 5.5, 3.5), (2.5, 5.5, 3.5)), cls.k)
        cls.history = PhysicalSamplingHistory(clock_id='physics-seconds')
        cls.history.add(SamplingView(frame, 'original', 'camera', 'physics-seconds', 0., .05), now_s=.05)

    def check(self, history=None, now=1., assumption=SamplingAssumption(.2, .2), ttl=20.):
        return (history or deepcopy(self.history)).check((4, 5), (5, 5), now_s=now,
                    budget=FlightBudget(free_ttl_s=ttl), assumption=assumption)

    def test_physical_volume_and_deadline_instead_of_kinematic_surrogate(self):
        r = self.check()
        self.assertTrue(r['allowed'])
        self.assertEqual(set(map(tuple, r['required_cells'])), spatial_cells((4, 5), (5, 5), .77))
        self.assertEqual(len(r['required_cells']), 12)
        self.assertEqual(r['baseline']['latest_stop_s'], 9.5)
        self.assertFalse(r['flight_authorized'])

    def test_expiry_is_old_capture_plus_lifetime(self):
        r = self.check(now=12.)
        self.assertFalse(r['allowed'])
        for row in r['sampling']:
            self.assertEqual(row['candidates'][0]['captured_at_s'], 0.)
            self.assertEqual(row['candidates'][0]['valid_until_s'], 20.)
            self.assertIn('SAMPLING_EXPIRES_BEFORE_PHYSICAL_STOP', row['candidates'][0]['reasons'])

    def test_unknown_size_never_cleared_by_coverage(self):
        r = self.check(assumption=None)
        self.assertTrue(r['baseline']['allowed'])
        self.assertFalse(r['allowed'])

    def test_current_occupancy_wins_even_if_old_sampling_is_supported(self):
        h = deepcopy(self.history)
        h.grid.occupied.add((5, 5, 3))
        r = self.check(h)
        self.assertFalse(r['allowed'])
        self.assertTrue(all(row['supported'] for row in r['sampling']))

    def test_consumer_clock_cannot_rewind(self):
        h = deepcopy(self.history)
        self.check(h, now=2.)
        with self.assertRaises(ValueError):
            self.check(h, now=1.)


@unittest.skipUnless(DEFAULT_DLL.exists(), 'verified native runtime unavailable')
class ColdStartTests(unittest.TestCase):
    def test_cold_start_fixture_really_removes_early_evidence(self):
        spec = next(s for s in specs() if s['key'] == 'no-history')
        self.assertFalse(spec['history'])
        _, r = simulate_case(spec)
        q = r['result']
        self.assertIsNone(q['receipt'])
        self.assertEqual(q['state'], 'SCAN_BUDGET_HOLD')
        self.assertTrue(all(not row['candidates'] for row in q['initial']['sampling']))
        self.assertTrue(all(c['frame_id'] != 'approach-history'
                            for row in q['final_guard']['sampling'] for c in row['candidates']))


@unittest.skipUnless(DEFAULT_DLL.exists(), 'verified native runtime unavailable')
class PhysicalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chosen = {'default-age', 'static-20s', 'thin-repeat', 'thin-diverse', 'depth-loss', 'scan-disturbance', 'ignored-command'}
        cls.results = {s['key']: simulate_case(s) for s in specs() if s['key'] in chosen}

    def test_actual_approach_precedes_rescan_and_motion_confirms_arrival(self):
        raw, r = self.results['static-20s']
        q = r['result']
        self.assertTrue(r['approach']['completed'])
        self.assertEqual(q['state'], 'ARRIVED_AND_SLOW')
        self.assertEqual(q['confirmed_cell'], [5, 5])
        self.assertTrue(q['receipt']['completed'])
        self.assertGreater(q['receipt']['position_error_m'], .0001)
        self.assertLessEqual(q['receipt']['position_error_m'], .03)
        self.assertEqual(q['actual_start'], r['approach']['end'])
        for item in raw['frames']:
            self.assertEqual(item['frame']['pose']['position'], item['capture_state']['position'])
        self.assertAlmostEqual(q['scan_duration_s'], .6)
        self.assertEqual(r['audit']['clearance_violations'], 0)

    def test_default_age_does_not_inherit_longer_static_fixture(self):
        _, r = self.results['default-age']
        self.assertEqual(r['budget']['free_ttl_s'], 12.)
        self.assertIsNone(r['result']['receipt'])
        self.assertEqual(len(r['actions']), 1)

    def test_rotation_prevents_specific_failure_without_removing_failed_repeat(self):
        _, repeat = self.results['thin-repeat']
        _, diverse = self.results['thin-diverse']
        self.assertGreater(repeat['audit']['clearance_violations'], 0)
        self.assertFalse(repeat['result']['receipt']['completed'])
        self.assertEqual(repeat['result']['confirmed_cell'], [4, 5])
        self.assertEqual(diverse['result']['state'], 'OBSERVED_OBSTACLE_HOLD')
        self.assertEqual(len(diverse['result']['events']), 2)
        self.assertIsNone(diverse['result']['receipt'])
        self.assertEqual(diverse['audit']['clearance_violations'], 0)

    def test_sensor_and_thrust_faults_do_not_issue_second_move(self):
        for key, state in [('depth-loss', 'SCAN_SENSOR_HOLD'), ('scan-disturbance', 'SCAN_STABILITY_HOLD')]:
            _, r = self.results[key]
            self.assertEqual(r['result']['state'], state)
            self.assertEqual(len(r['actions']), 1)
            self.assertIsNone(r['result']['receipt'])

    def test_ignored_command_does_not_advance_confirmed_cell(self):
        _, r = self.results['ignored-command']
        self.assertEqual(r['result']['state'], 'CONTROLLER_TIMEOUT')
        self.assertEqual(r['result']['confirmed_cell'], [4, 5])
        self.assertFalse(r['result']['receipt']['completed'])

    def test_saved_camera_frames_reproduce_control_and_final_state(self):
        raw, expected = self.results['static-20s']
        with patch('drone_nav.physical_vehicle.render', side_effect=AssertionError('replay must not render')):
            replay, result = simulate_case(expected['spec'], raw)
        self.assertEqual(replay, raw)
        self.assertEqual(result, expected)

    def test_invalid_capture_object_is_protocol_hold(self):
        v = PhysicalVehicle(world_for(specs()[1]), (4, 5))
        try:
            with patch.object(v, 'capture', return_value=None):
                r = rescan_and_move(v, PhysicalSamplingHistory(clock_id='physics-seconds'),
                                    (4, 5), (5, 5), Intrinsics(40, 30, 25, 25, 19.5, 14.5))
            self.assertEqual(r['state'], 'SCAN_PROTOCOL_HOLD')
            self.assertEqual(v.actions, [])
        finally:
            v.close()


if __name__ == '__main__':
    unittest.main()
