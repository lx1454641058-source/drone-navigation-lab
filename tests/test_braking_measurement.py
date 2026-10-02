"""来源：本项目原创。人工采样轨迹验证，测试不加载原生物理库。"""
import unittest
from math import sqrt
from drone_nav.braking_measurement import analyze, BrakeCase, DT


def trace(count=501, velocity=lambda t: (.1, 0., 0.), position=lambda t: (.1*t, 0., 3.5)):
    return [dict(time_s=i*DT, position=list(position(i*DT)), velocity=list(velocity(i*DT)),
                 quaternion=[1., 0., 0., 0.], angular_velocity=[0., 0., 0.]) for i in range(count)]


class BrakingMeasurementTests(unittest.TestCase):
    def test_constant_motion_exceeds_ideal_and_never_stabilizes(self):
        m = analyze(trace(), 0, (1., 0.))
        self.assertAlmostEqual(m['max_forward_m'], .1)
        self.assertAlmostEqual(m['ideal_distance_m'], .025)
        self.assertAlmostEqual(m['ideal_stop_time_s'], .3)
        self.assertIsNone(m['hold_window']['first_confirmation_s'])
        self.assertFalse(m['slow_from_ideal_deadline'])

    def test_short_slow_crossing_is_not_a_confirmed_stop(self):
        states = trace(velocity=lambda t: (0. if .1 <= t <= .2 else .1, 0., 0.))
        self.assertIsNone(analyze(states, 0, (1., 0.))['speed_window']['first_confirmation_s'])

    def test_return_to_speed_after_first_confirmation_is_retained(self):
        states = trace(velocity=lambda t: (0. if .1 <= t <= .5 or t >= .7 else .1, 0., 0.))
        window = analyze(states, 0, (1., 0.))['speed_window']
        self.assertAlmostEqual(window['first_confirmation_s'], .4)
        self.assertTrue(window['left_after_confirmation'])
        self.assertAlmostEqual(window['final_continuous_start_s'], .7)
        self.assertTrue(window['final_window_confirmed'])

    def test_slow_away_from_anchor_is_not_position_hold(self):
        states = trace(velocity=lambda t: (0., 0., 0.), position=lambda t: (.04 if t > 0 else 0., 0., 3.5))
        m = analyze(states, 0, (1., 0.))
        self.assertAlmostEqual(m['speed_window']['first_confirmation_s'], .3)
        self.assertIsNone(m['hold_window']['first_confirmation_s'])

    def test_negative_direction_measures_positive_forward_progress(self):
        m = analyze(trace(velocity=lambda t: (-.1, 0., 0.), position=lambda t: (-.1*t, 0., 3.5)), 0, (-1., 0.))
        self.assertAlmostEqual(m['max_forward_m'], .1)
        self.assertEqual(m['max_backward_m'], 0.)

    def test_diagonal_radial_and_lateral_are_distinct(self):
        states = trace(velocity=lambda t: (.1/sqrt(2), .1/sqrt(2), 0.),
                       position=lambda t: (.1*t/sqrt(2), .1*t/sqrt(2), 3.5))
        m = analyze(states, 0, (1/sqrt(2), 1/sqrt(2)))
        self.assertAlmostEqual(m['max_horizontal_radius_m'], .1)
        self.assertAlmostEqual(m['max_lateral_m'], 0.)

    def test_reverse_travel_is_not_erased_by_final_position(self):
        states = trace(position=lambda t: (.1*t if t <= .5 else .1*(1-t), 0., 3.5))
        m = analyze(states, 0, (1., 0.))
        self.assertAlmostEqual(m['max_forward_m'], .05)
        self.assertAlmostEqual(m['horizontal_path_m'], .1)
        self.assertEqual(m['final_offset_m'][0], 0.)

    def test_vertical_speed_prevents_false_stability(self):
        m = analyze(trace(velocity=lambda t: (0., 0., .04)), 0, (1., 0.))
        self.assertIsNone(m['speed_window']['first_confirmation_s'])
        self.assertFalse(m['entry_already_slow'])

    def test_bad_clock_and_nonfinite_feedback_rejected(self):
        for fault in ('gap', 'nan'):
            states = trace()
            if fault == 'gap':
                states[5]['time_s'] += DT
            else:
                states[5]['position'][0] = float('nan')
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                analyze(states, 0, (1., 0.))

    def test_case_timing_and_direction_rejected(self):
        for kwargs in (dict(control_seconds=.003), dict(direction=(1., 1.)), dict(delay_seconds=-1), dict(speed=float('nan'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BrakeCase('bad', **kwargs)

    def test_request_index_and_existing_tolerances(self):
        from drone_nav.physical_vehicle import FlightBudget
        from drone_nav.braking_measurement import SPEED_TOLERANCE, POSITION_TOLERANCE, HEIGHT_TOLERANCE, STABLE_SECONDS
        budget = FlightBudget()
        self.assertEqual((SPEED_TOLERANCE, POSITION_TOLERANCE, HEIGHT_TOLERANCE, STABLE_SECONDS),
                         (budget.speed_tolerance_mps, budget.position_tolerance_m, budget.height_tolerance_m, budget.stable_s))
        with self.assertRaises(ValueError):
            analyze(trace(), -1, (1., 0.))


if __name__ == '__main__':
    unittest.main()
