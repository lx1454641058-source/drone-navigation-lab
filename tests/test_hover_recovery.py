"""来源：本项目原创。连续稳定、异常反馈、任务状态保持与物理恢复测试。"""
from copy import deepcopy
from dataclasses import replace
from math import cos, sin, radians
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from drone_nav.hover_recovery import RecoveryBudget, recover_hover, finish_aborted_operation, rescan_with_recovery
from drone_nav.native_physics import DEFAULT_DLL
from tools.hover_recovery_experiment import read_parent, specs, run_case


class FakeVehicle:
    def __init__(self, speeds=None, angular=0., tilt=0., clock_fault=False):
        self.physics = SimpleNamespace(dt=.1)
        self.hold_target = (4.5, 5.5, 3.5)
        self.i = 0
        self.speeds = speeds or [0.]
        self.angular, self.tilt, self.clock_fault = angular, tilt, clock_fault
        self.targets = []

    def state(self):
        speed = self.speeds[min(self.i, len(self.speeds)-1)]
        angle = radians(self.tilt)/2
        return dict(time_s=self.i*.1, position=list(self.hold_target),
                    velocity=[speed, 0., 0.], quaternion=[cos(angle), sin(angle), 0., 0.],
                    angular_velocity=[self.angular, 0., 0.])

    def _step(self, target):
        self.targets.append(target)
        self.i += 1
        state = self.state()
        if self.clock_fault:
            state['time_s'] -= .1
        return state


class RecoveryContractTests(unittest.TestCase):
    def test_already_slow_still_requires_continuous_window(self):
        v = FakeVehicle()
        r = recover_hover(v)
        self.assertTrue(r['motion_stable'])
        self.assertEqual(len(v.targets), 3)
        self.assertAlmostEqual(r['stable_duration_s'], .3)
        self.assertTrue(all(t == (4.5, 5.5, 3.5) for t in v.targets))
        self.assertFalse(r['resume_allowed'])
        self.assertFalse(r['spatial_safety_certified'])

    def test_one_bad_sample_resets_accumulated_stable_time(self):
        v = FakeVehicle([0., 0., 0., .04, 0., 0., 0., 0.])
        r = recover_hover(v)
        self.assertEqual(v.i, 7)
        self.assertAlmostEqual(r['stable_duration_s'], .3)

    def test_low_linear_speed_does_not_hide_rotation_or_tilt(self):
        for v in (FakeVehicle(angular=.1), FakeVehicle(tilt=6.)):
            r = recover_hover(v, budget=RecoveryBudget(max_duration_s=.3))
            self.assertEqual(r['state'], 'RECOVERY_TIMEOUT')
            self.assertFalse(r['motion_stable'])

    def test_clock_fault_does_not_add_unverified_sample(self):
        r = recover_hover(FakeVehicle(clock_fault=True))
        self.assertEqual(r['state'], 'RECOVERY_CLOCK_FAULT')
        self.assertFalse(r['current_state_verified'])
        self.assertEqual(len(r['states']), 1)

    def test_bad_state_stays_unconfirmed(self):
        v = FakeVehicle()
        with patch.object(v, '_step', return_value={'time_s': .1, 'position': [float('nan'), 0., 0.]}):
            r = recover_hover(v)
        self.assertEqual(r['state'], 'RECOVERY_STATE_UNAVAILABLE')
        self.assertFalse(r['current_state_verified'])

    def test_initial_envelope_breach_sends_no_new_control(self):
        v = FakeVehicle()
        state = v.state(); state['position'][0] += .13
        with patch.object(v, 'state', return_value=state):
            r = recover_hover(v)
        self.assertEqual(r['state'], 'RECOVERY_ENVELOPE_BREACH')
        self.assertEqual(v.targets, [])

    def test_invalid_configuration_rejected_before_control(self):
        for kwargs in ({'max_duration_s': 0.}, {'stable_s': 5.}, {'position_error_m': .13}, {'tilt_deg': 90.}):
            with self.assertRaises(ValueError):
                RecoveryBudget(**kwargs)
        v = FakeVehicle()
        for b in (False, RecoveryBudget(stable_s=.35)):
            with self.assertRaises(ValueError):
                recover_hover(v, budget=b)
        self.assertEqual(v.targets, [])

    def test_recovery_keeps_original_failure_and_confirmation_cell(self):
        v = FakeVehicle()
        op = dict(state='SCAN_SENSOR_HOLD', actual_final=v.state(), confirmed_cell=[4, 5], receipt=None)
        r = finish_aborted_operation(v, op)
        self.assertEqual(r['state'], 'ABORTED_MOTION_STABLE')
        self.assertEqual(r['original_state'], 'SCAN_SENSOR_HOLD')
        self.assertEqual(r['confirmed_cell'], [4, 5])
        self.assertEqual(r['operation'], op)
        self.assertFalse(r['resume_allowed'])

    def test_success_skips_recovery_and_stale_receipt_is_rejected(self):
        v = FakeVehicle()
        op = dict(state='ARRIVED_AND_SLOW', actual_final=v.state(), receipt={'completed': True})
        r = finish_aborted_operation(v, op)
        self.assertIsNone(r['recovery'])
        self.assertEqual(v.targets, [])
        op['actual_final']['time_s'] = 1.
        with self.assertRaises(ValueError):
            finish_aborted_operation(v, op)

    def test_rescan_wrapper_consumes_final_receipt_before_recovery(self):
        v = FakeVehicle()
        op = dict(state='SCAN_SENSOR_HOLD', actual_final=v.state(), confirmed_cell=[4, 5], receipt=None)
        with patch('drone_nav.hover_recovery.rescan_and_move', return_value=op) as scan:
            r = rescan_with_recovery(v, object(), (4, 5), (5, 5), object())
        self.assertEqual(scan.call_count, 1)
        self.assertEqual(r['original_state'], 'SCAN_SENSOR_HOLD')
        self.assertTrue(r['recovery']['motion_stable'])


@unittest.skipUnless(DEFAULT_DLL.exists(), 'verified native physics runtime unavailable')
class NativeRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = read_parent()
        lookup = {c['key']:c for c in parent['cases']}
        cls.results = {s['key']:run_case(s, lookup[s['parent']]) for s in specs(parent)
                       if s['key'] in ('scan-disturbance', 'short-budget', 'persistent-thrust', 'thin-repeat')}

    def test_previous_unconfirmed_thrust_case_recovers_without_new_waypoint(self):
        raw, r = self.results['scan-disturbance']
        self.assertEqual(r['recovery']['state'], 'MOTION_STABLE')
        self.assertGreater(r['recovery']['elapsed_verified_s'], .5)
        self.assertEqual(r['control_actions'], 0)
        self.assertTrue(all(s['qualified'] for s in raw['samples'][-151:]))
        self.assertEqual(r['additional_clearance_intervals'], 0)

    def test_short_budget_or_persistent_fault_remains_failure(self):
        for key, state in [('short-budget', 'RECOVERY_TIMEOUT'), ('persistent-thrust', 'RECOVERY_ENVELOPE_BREACH')]:
            _, r = self.results[key]
            self.assertEqual(r['recovery']['state'], state)
            self.assertFalse(r['recovery']['motion_stable'])
            self.assertFalse(r['resume_allowed'])

    def test_contact_case_cannot_be_misreported_as_safe_or_successful(self):
        _, r = self.results['thin-repeat']
        self.assertGreater(r['final_audit']['clearance_violations'], 0)
        self.assertEqual(r['original_state'], 'CONTROLLER_TIMEOUT')
        self.assertFalse(r['recovery']['spatial_safety_certified'])
        self.assertFalse(r['flight_authorized'])
        self.assertEqual(r['confirmed_cell'], [4, 5])


if __name__ == '__main__':
    unittest.main()
