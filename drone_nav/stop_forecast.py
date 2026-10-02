"""来源：本项目原创。完整积分状态分支推演，仅供固定模型停止研究。"""
import ctypes as ct
from dataclasses import dataclass, asdict
import hashlib
import json
from math import hypot, isfinite
from pathlib import Path
from time import perf_counter

from .native_physics import QuadrotorPhysics, CTRL
from .flight_control import control
from .stopping_policy import AttitudeStop
from .braking_measurement import analyze, stable_window

INTEGRATION = (1 << 13)-1  # Verified against the fixed MuJoCo 3.3.7 mjdata.h.
CLOCK = 'mujoco-model-seconds'
WORLD = 'mujoco-quadrotor-world-m'


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode('utf-8')).hexdigest()


def integration_values(physics):
    physics._open()
    count = physics.lib.mj_stateSize(physics.model, INTEGRATION)
    if not 14 <= count <= 10000:
        raise ValueError('unexpected integration state size')
    values = (ct.c_double*count)()
    physics.lib.mj_getState(physics.model, physics.data, values, INTEGRATION)
    if not all(isfinite(v) for v in values):
        raise ValueError('nonfinite integration state')
    return list(values)


def implementation_digest():
    folder = Path(__file__).parent
    paths = ('stop_forecast.py', 'stopping_policy.py', 'flight_control.py', 'native_physics.py')
    return digest({name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in paths})


def binding(physics):
    return dict(clock=CLOCK, world=WORLD, state_sha256=digest(integration_values(physics)),
                model_sha256=physics.model_sha256, dll_sha256=physics.dll_sha256,
                implementation_sha256=implementation_digest())


class PhysicsBranch:
    """Own only a copied mjData; the parent owns the immutable model throughout use."""
    def __init__(self, parent):
        if type(parent) is not QuadrotorPhysics:
            raise TypeError('fixed QuadrotorPhysics instance required')
        parent._open()
        self.parent, self.model, self.lib = parent, parent.model, parent.lib
        self.dt = parent.dt
        self.data = None
        self.lib.mj_copyData.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
        self.lib.mj_copyData.restype = ct.c_void_p
        self.data = self.lib.mj_makeData(self.model)
        if not self.data:
            raise MemoryError('could not allocate forecast data')
        try:
            if self.lib.mj_copyData(self.data, self.model, parent.data) != self.data:
                raise ValueError('native data copy failed')
            if integration_values(self) != integration_values(parent):
                raise ValueError('branch initial integration state differs')
        except Exception:
            self.close()
            raise

    def _open(self):
        if not self.data or self.parent.model != self.model or not self.parent.data:
            raise RuntimeError('branch or owning model is closed')

    def state(self): return QuadrotorPhysics.state(self)
    def step(self, motors): return QuadrotorPhysics.step(self, motors)

    def close(self):
        if self.data:
            self.lib.mj_deleteData(self.data)
            self.data = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


@dataclass(frozen=True)
class StopRecipe:
    policy: str
    control_seconds: float = .002
    delay_seconds: float = 0.
    horizon_seconds: float = 8.
    direction: tuple = (1., 0.)

    def __post_init__(self):
        if self.policy not in ('original', 'attitude'):
            raise ValueError('unknown stop policy')
        values = (self.control_seconds, self.delay_seconds, self.horizon_seconds, *self.direction)
        if len(self.direction) != 2 or any(type(v) not in (int, float) or not isfinite(v) for v in values):
            raise ValueError('finite recipe required')
        if not .002 <= self.control_seconds <= .02 or not 0 <= self.delay_seconds <= .2 or self.horizon_seconds != 8.:
            raise ValueError('recipe outside fixed experiment range')
        if abs(hypot(*self.direction)-1) > 1e-9 or any(abs(v/.002-round(v/.002)) > 1e-8 for v in values[:3]):
            raise ValueError('unit direction and integral physics steps required')
        if not isinstance(self.direction, tuple):
            raise ValueError('immutable direction required')


@dataclass(frozen=True)
class FrozenForecast:
    content: str
    sha256: str

    @classmethod
    def create(cls, record):
        content = encode(record)
        return cls(content, hashlib.sha256(content.encode('utf-8')).hexdigest())

    def record(self):
        if hashlib.sha256(self.content.encode('utf-8')).hexdigest() != self.sha256:
            raise ValueError('forecast content digest mismatch')
        return json.loads(self.content)


def check_binding(forecast, current_binding, requested_recipe, *, clock_id):
    if type(forecast) is not FrozenForecast or type(requested_recipe) is not StopRecipe:
        raise TypeError('frozen forecast and stop recipe required')
    record = forecast.record()
    if clock_id != record['binding']['clock']:
        raise ValueError('clock mismatch')
    if record['binding'] != current_binding:
        raise ValueError('current state/model/implementation binding changed')
    if encode(record['recipe']) != encode(asdict(requested_recipe)):
        raise ValueError('stop recipe mismatch')
    return dict(binding_matches=True, execution_authorized=False,
                reason='MODEL_BINDING_ONLY_NO_SPATIAL_OR_ONLINE_AUTHORIZATION')


def rollout(physics, recipe, last_motors, *, fault=None):
    """Advance the passed instance. Disturbance injection is for separate actual trials."""
    if fault not in (None, 'pitch_pulse', 'thrust_loss'):
        raise ValueError('unknown experiment fault')
    if len(last_motors) != 4 or any(type(v) not in (float, int) or not isfinite(v) or not 0 <= v <= 8 for v in last_motors):
        raise ValueError('four valid prior motors required')
    entry = physics.state()
    states, commands = [entry], []
    anchor = tuple(entry['position'])
    controller = AttitudeStop(entry) if recipe.policy == 'attitude' else None
    delay, stride = round(recipe.delay_seconds/physics.dt), round(recipe.control_seconds/physics.dt)
    motors, target = list(last_motors), None
    hold_target = hold_index = None
    for i in range(round(recipe.horizon_seconds/physics.dt)):
        info = None
        updated = i >= delay and (i-delay) % stride == 0
        if updated:
            if controller is None:
                target = anchor
                motors, inner = control(states[-1], target)
                info = dict(mode='ORIGINAL_CONTROL', saturated=inner['saturated'])
                if hold_index is None: hold_target, hold_index = anchor, i
            else:
                motors, info = controller.command(states[-1])
                target = info['target']
                if controller.hold_target is not None and hold_index is None:
                    hold_target, hold_index = controller.hold_target, i
        actual = list(motors)
        if fault == 'pitch_pulse' and i < 100:
            actual = [max(0., min(8., m+delta)) for m, delta in zip(actual, (-.25, -.25, .25, .25))]
        elif fault == 'thrust_loss' and i < 100:
            actual = [m*.85 for m in actual]
        # Emit JSON-native values so live execution and frozen prediction share one schema.
        commands.append(dict(updated=updated, target=None if target is None else list(target), motors=actual, info=info))
        physics.step(actual)
        states.append(physics.state())
    times = [s['time_s']-entry['time_s'] for s in states]
    valid = [hold_index is not None and i >= hold_index and hypot(*s['velocity']) <= .03
             and hypot(*(p-a for p, a in zip(s['position'], hold_target))) <= .03
             and abs(s['position'][2]-hold_target[2]) <= .1 for i, s in enumerate(states)]
    stop = dict(hold_target=None if hold_target is None else list(hold_target),
                hold_latched_at_s=None if hold_index is None else times[hold_index],
                actual_hold_window=stable_window(times, valid))
    lower = [min(s['position'][j] for s in states) for j in range(3)]
    upper = [max(s['position'][j] for s in states) for j in range(3)]
    envelope = dict(center_lower_m=lower, center_upper_m=upper,
                    body_radius_m=.35, extra_margin_m=.02,
                    lower_m=[x-.37 for x in lower], upper_m=[x+.37 for x in upper],
                    horizon_end_s=states[-1]['time_s'], continuous_space_certified=False,
                    uncertainty_bound_validated=False)
    return dict(states=states, commands=commands, stop=stop, envelope=envelope,
                metrics=analyze(states, 0, recipe.direction), request_index=0,
                case=dict(direction=list(recipe.direction)), flight_authorized=False)


def forecast_stop(parent, recipe):
    if type(recipe) is not StopRecipe:
        raise TypeError('stop recipe required')
    started = perf_counter()
    initial_binding = binding(parent)
    controls = (ct.c_double*4)()
    parent.lib.mj_getState(parent.model, parent.data, controls, CTRL)
    with PhysicsBranch(parent) as branch:
        predicted = rollout(branch, recipe, list(controls))
    if binding(parent) != initial_binding:
        raise ValueError('prediction changed source integration state')
    record = dict(schema='stop-forecast-v1', binding=initial_binding, recipe=asdict(recipe),
                  last_motors=list(controls), prediction=predicted, flight_authorized=False)
    frozen = FrozenForecast.create(record)
    elapsed = perf_counter()-started
    return frozen, elapsed
