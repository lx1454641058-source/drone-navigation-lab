"""来源：本项目原创。运动学独立差分、控制映射与持续反馈条件。"""
from copy import deepcopy
from math import sin, cos, hypot
import unittest
from drone_nav.flight_control import euler_quaternion, quaternion_product
from drone_nav.stopping_policy import AttitudeStop, tilt_response, KV, KA, KJ


def state(t=0., velocity=(.2, 0., 0.), q=(1., 0., 0., 0.), omega=(0., 0., 0.)):
    return dict(time_s=t, position=[0., 0., 3.5], velocity=list(velocity), quaternion=list(q), angular_velocity=list(omega))


class StoppingPolicyTests(unittest.TestCase):
    def test_gains_match_declared_cubic(self):
        self.assertAlmostEqual(8.8+32*KJ, 18)
        self.assertAlmostEqual(32*(1+KA), 108)
        self.assertAlmostEqual(32*KV, 216)

    def test_zero_tilt_matches_velocity_damping_and_virtual_target(self):
        s = state()
        motors, info = AttitudeStop(s).command(s)
        self.assertAlmostEqual(info['requested_acceleration_mps2'][0], -1.35)
        self.assertAlmostEqual(1.8*info['target'][0]-2.8*.2, -1.35)
        self.assertTrue(all(0 <= m <= 8 for m in motors))
        self.assertFalse(info['target_is_navigation_waypoint'])

    def test_tilt_rate_matches_quaternion_finite_difference(self):
        q, omega = euler_quaternion(.12, -.08, .4), (.21, -.17, .11)
        s = state(q=q, omega=omega)
        _, rate = tilt_response(s)
        eps = 1e-6
        length = hypot(*omega)
        def tilt_at(sign):
            half = sign*eps*length/2
            delta = (cos(half), *(sin(half)*v/length for v in omega))
            rotated = quaternion_product(q, delta)
            # Rotate body z via q * z * conjugate(q), independent of matrix formula.
            v = quaternion_product(quaternion_product(rotated, (0., 0., 0., 1.)),
                                   (rotated[0], *(-x for x in rotated[1:])))
            return [9.81*v[i]/v[3] for i in (1, 2)]
        left, right = tilt_at(-1), tilt_at(1)
        for j in range(2):
            self.assertAlmostEqual(rate[j], (right[j]-left[j])/(2*eps), places=7)

    def test_quaternion_sign_does_not_change_estimate(self):
        q = euler_quaternion(.1, .05, .3)
        self.assertEqual(tilt_response(state(q=q, omega=(.2, .1, .3))),
                         tilt_response(state(q=tuple(-v for v in q), omega=(.2, .1, .3))))

    def test_stationary_but_tilted_or_rotating_does_not_latch(self):
        for q, omega in ((euler_quaternion(0, .1), (0, 0, 0)), ((1, 0, 0, 0), (0, .1, 0))):
            controller = AttitudeStop(state(velocity=(0, 0, 0), q=q, omega=omega))
            for i in range(21):
                controller.command(state(t=i*.02, velocity=(0, 0, 0), q=q, omega=omega))
            self.assertIsNone(controller.hold_target)

    def test_quiet_window_latches_actual_position_and_never_retargets(self):
        controller = AttitudeStop(state(velocity=(0, 0, 0)))
        for i in range(16):
            s = state(t=i*.02, velocity=(0, 0, 0))
            s['position'][0] = .1
            controller.command(s)
        self.assertEqual(controller.hold_target, (.1, 0, 3.5))
        later = state(.32)
        later['position'][0] = .15
        _, info = controller.command(later)
        self.assertEqual(info['target'], [.1, 0, 3.5])

    def test_gap_cannot_count_unobserved_time_as_quiet(self):
        controller = AttitudeStop(state(velocity=(0, 0, 0)))
        controller.command(state(0, velocity=(0, 0, 0)))
        controller.command(state(.4, velocity=(0, 0, 0)))
        self.assertIsNone(controller.hold_target)
        self.assertEqual(controller.quiet_since, .4)

    def test_motion_resets_quiet_window(self):
        controller = AttitudeStop(state(velocity=(0, 0, 0)))
        for i in range(11):
            controller.command(state(i*.02, velocity=(0, 0, 0)))
        controller.command(state(.22))
        self.assertIsNone(controller.quiet_since)
        controller.command(state(.24, velocity=(0, 0, 0)))
        self.assertEqual(controller.quiet_since, .24)

    def test_invalid_feedback_latches_without_committing_clock(self):
        for fault in ('nan', 'quaternion', 'tilt', 'clock'):
            controller = AttitudeStop(state())
            controller.command(state())
            bad = state(.02)
            if fault == 'nan': bad['velocity'][0] = float('nan')
            elif fault == 'quaternion': bad['quaternion'][0] = 2.
            elif fault == 'tilt': bad['quaternion'] = list(euler_quaternion(0, 1.))
            else: bad['time_s'] = 0.
            with self.subTest(fault=fault), self.assertRaises(ValueError): controller.command(bad)
            self.assertEqual(controller.last_time, 0.)
            with self.assertRaises(ValueError): controller.command(state(.04))

    def test_high_request_is_clipped_and_reported(self):
        s = state(velocity=(1., -1., 0.))
        _, info = AttitudeStop(s).command(s)
        self.assertEqual(info['requested_acceleration_mps2'], [-2.5, 2.5])
        self.assertTrue(info['acceleration_clipped'])


if __name__ == '__main__':
    unittest.main()
