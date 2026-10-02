"""来源：本项目原创。移动状态交付的子范围、状态、坐标及双时钟边界。"""
from dataclasses import replace
import unittest

from drone_nav.following_space import prepare_following_space, consume_following_space
from drone_nav.stop_forecast import FrozenForecast, CLOCK
import test_forecast_space as fixture_module


class FollowingSpaceTests(unittest.TestCase):
    def fixture(self, z=5.):
        monitor, current, inspector, reference, now = fixture_module.ForecastSpaceTests().fixture(z)
        receipt = prepare_following_space(monitor, current(100), inspector, reference, now_s=now, clock_id=CLOCK)
        return monitor, current, inspector, reference, receipt

    def consume(self, values, index=110, **overrides):
        m, c, i, r, receipt = values
        m.tick(c(index), elapsed_s=index*.002)
        args = dict(now_s=c(index)['time_s'], clock_id=CLOCK, wall_age_s=.12)
        args.update(overrides)
        return consume_following_space(receipt, m, c(index), r, **args)

    def test_advanced_matching_state_retains_diagnostic_and_all_limits(self):
        result = self.consume(self.fixture())
        self.assertTrue(result['diagnostic_current'])
        self.assertTrue(result['remaining_subset'])
        self.assertGreater(result['current_index'], result['query_index'])
        for key in ('flight_authorized', 'selected_for_execution', 'free_volume_proven', 'navigation_map_update_allowed'):
            self.assertFalse(result[key])
        self.assertIn('STOP_MODEL_UNCERTAINTY_UNVALIDATED', result['reasons'])
        self.assertIn('OBSERVATION_EXPIRES_BEFORE_FORECAST_END', result['reasons'])

    def test_superset_surface_restriction_is_not_erased_by_motion(self):
        result = self.consume(self.fixture(2.))
        self.assertTrue(result['diagnostic_current'])
        self.assertEqual(result['checked_superset_geometry'], 'MEASURED_SURFACE')
        self.assertFalse(result['flight_authorized'])

    def test_different_current_controller_is_rejected_even_if_monitor_cache_matches(self):
        m, c, i, r, q = self.fixture()
        m.tick(c(110), elapsed_s=.22)
        altered = dict(c(110), controller_sha256='d'*64)
        result = consume_following_space(q, m, altered, r, now_s=altered['time_s'], clock_id=CLOCK, wall_age_s=.12)
        self.assertIn('CURRENT_CHECKPOINT_UNPROVEN', result['delivery_failures'])
        self.assertFalse(result['diagnostic_current'])

    def test_monitor_divergence_and_deadline_remain_latched(self):
        for kind in ('divergence', 'deadline'):
            with self.subTest(kind=kind):
                f = self.fixture(); m, c, i, r, q = f
                m.tick(dict(c(110), state_sha256='f'*64) if kind == 'divergence' else c(110),
                       elapsed_s=.22, deadline_missed=kind == 'deadline')
                result = self.consume(f, index=120)
                self.assertFalse(result['diagnostic_current'])
                self.assertTrue(any(reason.startswith('STOP_PREDICTION_NOT_MATCHED:') for reason in result['delivery_failures']))

    def test_wall_age_rejects_even_when_model_has_not_caught_up(self):
        result = self.consume(self.fixture(), wall_age_s=.50001)
        self.assertIn('DEPTH_EXPIRED_ON_WALL_CLOCK', result['delivery_failures'])
        self.assertNotIn('DEPTH_EXPIRED_ON_MODEL_CLOCK', result['delivery_failures'])

    def test_model_age_rejects_even_with_small_wall_age(self):
        f = self.fixture(); m, c, i, r, q = f
        for index in range(110, 310, 10):
            m.tick(c(index), elapsed_s=index*.002)
        result = self.consume(f, index=310)
        self.assertIn('DEPTH_EXPIRED_ON_MODEL_CLOCK', result['delivery_failures'])
        self.assertFalse(result['diagnostic_current'])

    def test_unrelated_clocks_are_not_used_for_expiry(self):
        result = self.consume(self.fixture(), clock_id='foreign', now_s=1000.,
                              wall_clock_id='foreign-wall', wall_age_s=1000.)
        self.assertIn('COMPLETION_CLOCK_MISMATCH', result['delivery_failures'])
        self.assertIn('WALL_CLOCK_MISMATCH', result['delivery_failures'])
        self.assertNotIn('DEPTH_EXPIRED_ON_MODEL_CLOCK', result['delivery_failures'])
        self.assertNotIn('DEPTH_EXPIRED_ON_WALL_CLOCK', result['delivery_failures'])

    def test_reference_and_prediction_identity_changes_are_rejected(self):
        m, c, i, r, q = self.fixture()
        m.tick(c(110), elapsed_s=.22)
        result = consume_following_space(q, m, c(110), replace(r, reference_fingerprint='f'*64),
            now_s=c(110)['time_s'], clock_id=CLOCK, wall_age_s=.12)
        self.assertIn('REFERENCE_CHANGED', result['delivery_failures'])
        changed = m.packet.record(); changed['request_id'] += '-other'
        m.packet = FrozenForecast.create(changed)
        result = consume_following_space(q, m, c(110), r, now_s=c(110)['time_s'], clock_id=CLOCK, wall_age_s=.12)
        self.assertIn('PREDICTION_IDENTITY_CHANGED', result['delivery_failures'])

    def test_queried_superset_cannot_be_silently_enlarged(self):
        f = list(self.fixture()); sealed = f[-1].record()
        sealed['proofs'][1]['upper'][0] += .001
        f[-1] = FrozenForecast.create(sealed)
        result = self.consume(f)
        self.assertIn('REMAINING_VOLUME_OUTSIDE_CHECKED_REGION', result['delivery_failures'])

    def test_depth_is_a_sealed_snapshot_not_a_mutable_buffer(self):
        f = self.fixture(); f[2].depths = (100.,)*25
        result = self.consume(f)
        self.assertTrue(result['diagnostic_current'])
        # The original query remains bound to its original depth identity.
        self.assertIsNotNone(FrozenForecast(**f[-1].record()['query']).record()['depth_identity'])

    def test_bad_time_and_tampered_receipt_fail_explicitly(self):
        for age in (-1., float('nan')):
            with self.assertRaises(ValueError):
                self.consume(self.fixture(), wall_age_s=age)
        f = list(self.fixture()); f[-1] = replace(f[-1], content=f[-1].content+' ')
        with self.assertRaises(ValueError): self.consume(f)

    def test_current_time_index_and_unmatched_query_cannot_pass(self):
        result = self.consume(self.fixture(), now_s=7.21)
        self.assertIn('CURRENT_TIME_MISMATCH', result['delivery_failures'])
        f = self.fixture(); m,c,i,r,q=f
        m.tick(c(110), elapsed_s=.22, deadline_missed=True)
        rejected = prepare_following_space(m,c(110),i,r,now_s=c(110)['time_s'],clock_id=CLOCK)
        result=consume_following_space(rejected,m,c(110),r,now_s=c(110)['time_s'],clock_id=CLOCK,wall_age_s=.12)
        self.assertIn('NO_GEOMETRY_QUERY',result['delivery_failures'])
        self.assertFalse(result['diagnostic_current'])


if __name__ == '__main__':
    unittest.main()
