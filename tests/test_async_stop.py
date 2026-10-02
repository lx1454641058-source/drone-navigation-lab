"""来源：本项目原创。异步结果的时序、状态记忆和失效锁定。原生并发另行验证。"""
from dataclasses import asdict, replace
from copy import deepcopy
import unittest

from drone_nav.async_stop import (StopStepper, ForecastMonitor, implementation, RESULT_DEADLINE,
                                 HORIZON_STEPS, TICK_STEPS)
from drone_nav.stop_forecast import FrozenForecast, StopRecipe, digest, CLOCK, WORLD


class AsyncStopTests(unittest.TestCase):
    def fixture(self):
        entry = dict(time_s=7., position=[0., 0., 3.5], quaternion=[1., 0., 0., 0.],
                     velocity=[0., 0., 0.], angular_velocity=[0., 0., 0.])
        recipe = StopRecipe('attitude', control_seconds=.02)
        state = [7., 0., 0., 3.5, 1., 0., 0., 0., *([0.]*6)]
        runtime = dict(clock=CLOCK, world=WORLD, model_sha256='fixed-model', dll_sha256='fixed-dll')
        record = dict(schema='async-stop-request-v1', request_id='one', integration=state,
            binding=dict(**runtime, state_sha256=digest(state), implementation_sha256='fixed-parent'),
            implementation=implementation(), entry=entry, prior_motors=[3.4335]*4, recipe=asdict(recipe))
        request = FrozenForecast.create(record)
        points = []
        for i in range(0, HORIZON_STEPS+1, TICK_STEPS):
            points.append(dict(index=i, time_s=7.+i*.002, **runtime, implementation=record['implementation'],
                state_sha256=digest(['physics', i]), controller_sha256=digest(['controller', i]),
                remaining=dict(center_lower_m=[0., 0., 3.5], center_upper_m=[0., 0., 3.5],
                               lower_m=[-.37, -.37, 3.5-.37], upper_m=[.37, .37, 3.5+.37],
                               through_step=HORIZON_STEPS, continuous_space_certified=False,
                               uncertainty_bound_validated=False)))
        packet = FrozenForecast.create(dict(schema='async-stop-index-v1', request_sha256=request.sha256,
            request_id='one', recipe=request.record()['recipe'], checkpoints=points, execution_authorized=False))
        current = lambda i: {k:v for k,v in points[i//TICK_STEPS].items() if k != 'remaining'}
        return request, packet, current

    def test_delayed_result_matches_current_time_not_original_start(self):
        request, packet, current = self.fixture()
        monitor = ForecastMonitor(request)
        for i in range(0, 100, 10):
            self.assertEqual(monitor.tick(current(i), elapsed_s=i*.002)['reason'], 'WAITING')
        value = monitor.tick(current(100), elapsed_s=.2, packet=packet)
        self.assertTrue(value['forecast_matches'])
        self.assertEqual(value['accepted_at_step'], 100)
        self.assertFalse(value['execution_authorized'])

    def test_all_runtime_and_state_fields_are_bound(self):
        request, packet, current = self.fixture()
        for key in ('time_s', 'clock', 'world', 'model_sha256', 'dll_sha256', 'implementation',
                    'state_sha256', 'controller_sha256'):
            with self.subTest(key=key):
                monitor = ForecastMonitor(request)
                bad = {**current(100), key: 'different'}
                self.assertEqual(monitor.tick(bad, elapsed_s=.2, packet=packet)['reason'], 'EXECUTION_STATE_DIVERGED')

    def test_divergence_is_latched_even_if_later_state_matches(self):
        request, packet, current = self.fixture()
        monitor = ForecastMonitor(request)
        monitor.tick(current(100), elapsed_s=.2, packet=packet)
        bad = dict(current(110), controller_sha256=digest('new-hold-point'))
        self.assertEqual(monitor.tick(bad, elapsed_s=.22)['reason'], 'EXECUTION_STATE_DIVERGED')
        result = monitor.tick(current(120), elapsed_s=.24)
        self.assertFalse(result['forecast_matches'])
        self.assertIsNone(result['remaining'])

    def test_timeout_cannot_be_cleared_by_late_result(self):
        request, packet, current = self.fixture()
        monitor = ForecastMonitor(request)
        self.assertEqual(monitor.tick(current(260), elapsed_s=.52)['reason'], 'RESULT_TIMEOUT')
        self.assertEqual(monitor.tick(current(270), elapsed_s=.54, packet=packet)['reason'], 'RESULT_TIMEOUT')
        self.assertIsNone(monitor.packet)

    def test_deadline_boundary_and_first_late_delivery(self):
        request, packet, current = self.fixture()
        self.assertTrue(ForecastMonitor(request).tick(current(250), elapsed_s=RESULT_DEADLINE, packet=packet)['forecast_matches'])
        self.assertEqual(ForecastMonitor(request).tick(current(250), elapsed_s=RESULT_DEADLINE+1e-9, packet=packet)['reason'], 'RESULT_TIMEOUT')

    def test_wrong_request_recipe_and_tampered_payload_rejected(self):
        request, packet, current = self.fixture()
        variants = [replace(packet, content=packet.content+' ')]
        for field, value in (('request_id', 'other'), ('request_sha256', digest('other')),
                             ('recipe', asdict(StopRecipe('original', control_seconds=.02))),
                             ('execution_authorized', True)):
            variants.append(FrozenForecast.create({**packet.record(), field:value}))
        for value in variants:
            self.assertEqual(ForecastMonitor(request).tick(current(0), elapsed_s=0, packet=value)['reason'], 'RESULT_BINDING_INVALID')

    def test_repeated_result_and_sequence_gap_latch(self):
        request, packet, current = self.fixture()
        monitor = ForecastMonitor(request)
        monitor.tick(current(0), elapsed_s=0, packet=packet)
        self.assertEqual(monitor.tick(current(10), elapsed_s=.02, packet=packet)['reason'], 'REPEATED_RESULT')
        monitor = ForecastMonitor(request)
        monitor.tick(current(0), elapsed_s=0)
        self.assertEqual(monitor.tick(current(20), elapsed_s=.04, packet=packet)['reason'], 'MONITOR_SEQUENCE_ERROR')

    def test_worker_failure_control_overrun_and_wall_reversal(self):
        request, packet, current = self.fixture()
        for kwargs, reason in ((dict(worker_error=True), 'WORKER_FAILED'),
                               (dict(deadline_missed=True), 'CONTROL_DEADLINE_MISSED')):
            monitor = ForecastMonitor(request)
            self.assertEqual(monitor.tick(current(0), elapsed_s=0, packet=packet, **kwargs)['reason'], reason)
            self.assertFalse(monitor.tick(current(10), elapsed_s=.02)['forecast_matches'])
        monitor = ForecastMonitor(request)
        monitor.tick(current(0), elapsed_s=.2)
        self.assertEqual(monitor.tick(current(10), elapsed_s=.1, packet=packet)['reason'], 'WALL_CLOCK_REVERSED')

    def test_horizon_has_no_automatic_extension(self):
        request, packet, current = self.fixture()
        monitor = ForecastMonitor(request)
        monitor.tick(current(3990), elapsed_s=.2, packet=packet)
        value = monitor.tick(current(4000), elapsed_s=.22)
        self.assertEqual(value['reason'], 'HORIZON_EXHAUSTED')
        self.assertIsNone(value['remaining'])

    def test_truncated_or_nonfinite_or_invalid_envelope_rejected(self):
        request, packet, current = self.fixture()
        for mode in ('truncated', 'unordered', 'radius', 'claim'):
            value = packet.record()
            if mode == 'truncated': value['checkpoints'].pop()
            if mode == 'unordered': value['checkpoints'][1]['index'] = 0
            if mode == 'radius': value['checkpoints'][1]['remaining']['lower_m'][0] = 0
            if mode == 'claim': value['checkpoints'][1]['remaining']['continuous_space_certified'] = True
            result = ForecastMonitor(request).tick(current(0), elapsed_s=0, packet=FrozenForecast.create(value))
            self.assertEqual(result['reason'], 'RESULT_BINDING_INVALID')

    def test_controller_memory_is_not_just_pose_and_speed(self):
        request, _, _ = self.fixture()
        record = request.record()
        recipe = StopRecipe('attitude', control_seconds=.02)
        controller = StopStepper(record['entry'], recipe, record['prior_motors'])
        baseline = digest(controller.context())
        controller.controller.quiet_since = 7.
        self.assertNotEqual(digest(controller.context()), baseline)
        controller.controller.quiet_since = None
        controller.anchor = (1., 0., 3.5)
        self.assertNotEqual(digest(controller.context()), baseline)

    def test_feedback_clock_and_schedule_validation(self):
        request, _, _ = self.fixture()
        record = request.record()
        with self.assertRaises(ValueError): StopStepper(record['entry'], StopRecipe('original'), record['prior_motors'])
        controller = StopStepper(record['entry'], StopRecipe('original', control_seconds=.02), record['prior_motors'])
        controller.command(record['entry'])
        with self.assertRaises(ValueError): controller.command(record['entry'])


if __name__ == '__main__': unittest.main()
