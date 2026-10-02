"""来源：本项目原创。纯 Python 绑定与不可变内容测试；原生复制由完整实验验证。"""
from dataclasses import FrozenInstanceError, replace
import unittest
from drone_nav.stop_forecast import StopRecipe, FrozenForecast, check_binding, digest, CLOCK, WORLD


class StopForecastTests(unittest.TestCase):
    def fixture(self):
        recipe = StopRecipe('attitude')
        binding = dict(clock=CLOCK, world=WORLD, state_sha256=digest([1., 2., 3.]),
                       model_sha256='model', dll_sha256='dll', implementation_sha256='impl')
        from dataclasses import asdict
        forecast = FrozenForecast.create(dict(binding=binding, recipe=asdict(recipe), prediction=dict(states=[[1, 2, 3]])))
        return recipe, binding, forecast

    def test_matching_state_is_not_execution_authorization(self):
        r, b, f = self.fixture()
        result = check_binding(f, b, r, clock_id=CLOCK)
        self.assertTrue(result['binding_matches'])
        self.assertFalse(result['execution_authorized'])

    def test_record_is_an_independent_copy(self):
        _, _, f = self.fixture()
        record = f.record()
        record['prediction']['states'][0][0] = 9
        self.assertEqual(f.record()['prediction']['states'][0][0], 1)
        with self.assertRaises(FrozenInstanceError): f.content = '{}'

    def test_changed_payload_is_rejected(self):
        _, _, f = self.fixture()
        with self.assertRaises(ValueError): replace(f, content=f.content+' ').record()

    def test_policy_delay_and_period_are_bound(self):
        r, b, f = self.fixture()
        for changed in (replace(r, policy='original'), replace(r, delay_seconds=.2), replace(r, control_seconds=.02)):
            with self.subTest(recipe=changed), self.assertRaises(ValueError):
                check_binding(f, b, changed, clock_id=CLOCK)

    def test_state_model_clock_world_and_code_are_bound(self):
        r, b, f = self.fixture()
        for key in b:
            changed = dict(b, **{key: 'wrong'})
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_binding(f, changed, r, clock_id=CLOCK)
        with self.assertRaises(ValueError): check_binding(f, b, r, clock_id='wall-clock')

    def test_state_digest_includes_velocity_and_control_values(self):
        state = [0.]*42
        for index in (0, 7, 13, 20, 30):
            changed = list(state); changed[index] = .01
            self.assertNotEqual(digest(state), digest(changed))

    def test_live_rollout_and_frozen_record_use_identical_types(self):
        import json
        from copy import deepcopy
        from drone_nav.stop_forecast import rollout
        class StationaryPhysics:
            dt = .002
            steps = 0
            def state(self):
                return dict(time_s=self.steps*self.dt, position=[0., 0., 3.5], velocity=[0., 0., 0.],
                            quaternion=[1., 0., 0., 0.], angular_velocity=[0., 0., 0.])
            def step(self, motors): self.steps += 1
        for policy in ('original', 'attitude'):
            raw = rollout(StationaryPhysics(), StopRecipe(policy), [3.4335]*4)
            self.assertEqual(raw, FrozenForecast.create(raw).record())
            self.assertIsInstance(raw['stop']['hold_target'], list)

    def test_invalid_recipe_timings_and_direction_rejected(self):
        for kwargs in (dict(policy='unknown'), dict(control_seconds=.003), dict(delay_seconds=-.1),
                       dict(horizon_seconds=0), dict(direction=(1., 1.)), dict(direction=[1., 0.]),
                       dict(control_seconds=float('nan'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StopRecipe(**{'policy': 'original', **kwargs})


if __name__ == '__main__': unittest.main()
