"""来源：本项目原创。持续停止反馈与按时间索引的异步预测核对；不发导航许可。"""
import ctypes as ct
from dataclasses import asdict
import hashlib
from math import hypot, isfinite
from pathlib import Path

from .native_physics import QuadrotorPhysics, CTRL
from .flight_control import control
from .stopping_policy import AttitudeStop, validate_state
from .braking_measurement import analyze, stable_window
from .stop_forecast import (FrozenForecast, StopRecipe, INTEGRATION, CLOCK, WORLD,
                            binding, digest, integration_values, implementation_digest)

DT = .002
TICK_STEPS = 10
TICK_SECONDS = .02
HORIZON_STEPS = 4000
RESULT_DEADLINE = .5


def implementation():
    return digest(dict(parent=implementation_digest(),
                       adapter=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def make_request(physics, recipe, request_id):
    if type(physics) is not QuadrotorPhysics or type(recipe) is not StopRecipe:
        raise TypeError('fixed physics and recipe required')
    if recipe.control_seconds != TICK_SECONDS or recipe.delay_seconds != 0:
        raise ValueError('this async experiment requires 20 ms control and no stop delay')
    if not isinstance(request_id, str) or not request_id or len(request_id) > 100:
        raise ValueError('bounded nonempty request identity required')
    prior = (ct.c_double*4)()
    physics.lib.mj_getState(physics.model, physics.data, prior, CTRL)
    return FrozenForecast.create(dict(schema='async-stop-request-v1', request_id=request_id,
        binding=binding(physics), implementation=implementation(), integration=integration_values(physics),
        entry=physics.state(), prior_motors=list(prior), recipe=asdict(recipe)))


def request_record(request):
    if type(request) is not FrozenForecast:
        raise TypeError('immutable request required')
    record = request.record()
    if record['schema'] != 'async-stop-request-v1' or record['implementation'] != implementation():
        raise ValueError('request schema or implementation changed')
    recipe = StopRecipe(**{**record['recipe'], 'direction': tuple(record['recipe']['direction'])})
    if recipe.control_seconds != TICK_SECONDS or recipe.delay_seconds != 0:
        raise ValueError('unsupported asynchronous schedule')
    if digest(record['integration']) != record['binding']['state_sha256']:
        raise ValueError('request integration digest mismatch')
    validate_state(record['entry'])
    return record, recipe


def restore_request(physics, request):
    """Restore only the public integration vector into a separately compiled fixed model.

    mj_step computes forward dynamics before integrating. Calling mj_forward here
    could change saved solver warm-start state; no derived geometry is consumed.
    """
    record, recipe = request_record(request)
    runtime = binding(physics)
    for key in ('clock', 'world', 'model_sha256', 'dll_sha256', 'implementation_sha256'):
        if runtime[key] != record['binding'][key]:
            raise ValueError('worker runtime differs: '+key)
    values = record['integration']
    count = physics.lib.mj_stateSize(physics.model, INTEGRATION)
    if len(values) != count or any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise ValueError('invalid integration vector')
    physics.lib.mj_setState(physics.model, physics.data, (ct.c_double*count)(*values), INTEGRATION)
    if integration_values(physics) != values or physics.state() != record['entry']:
        raise ValueError('restored request differs')
    prior = (ct.c_double*4)()
    physics.lib.mj_getState(physics.model, physics.data, prior, CTRL)
    if list(prior) != record['prior_motors']:
        raise ValueError('prior motor controls differ')
    return StopStepper(record['entry'], recipe, record['prior_motors'])


class StopStepper:
    """Incremental version of the frozen stopping rollout, including controller memory."""
    def __init__(self, entry, recipe, prior_motors):
        validate_state(entry)
        if type(recipe) is not StopRecipe or recipe.control_seconds != TICK_SECONDS or recipe.delay_seconds != 0:
            raise ValueError('20 ms zero-delay stop recipe required')
        if len(prior_motors) != 4 or any(type(v) not in (int, float) or not isfinite(v) or not 0 <= v <= 8 for v in prior_motors):
            raise ValueError('four finite bounded prior motors required')
        self.recipe = recipe
        self.start = entry['time_s']
        self.anchor = tuple(entry['position'])
        self.motors = list(prior_motors)
        self.controller = AttitudeStop(entry) if recipe.policy == 'attitude' else None
        self.target = self.hold_target = self.hold_index = None
        self.index = 0
        self.implementation = implementation()

    def context(self):
        # The physical state alone omits the anchor, held motors and quiet/hold latches.
        return dict(index=self.index, start=self.start, anchor=self.anchor, recipe=asdict(self.recipe),
                    motors=self.motors, target=self.target, hold_target=self.hold_target,
                    hold_index=self.hold_index,
                    controller=None if self.controller is None else dict(vars(self.controller)))

    def command(self, state):
        validate_state(state)
        if self.index >= HORIZON_STEPS or abs(state['time_s']-(self.start+self.index*DT)) > 1e-8:
            raise ValueError('feedback clock differs or horizon exhausted')
        updated = self.index % TICK_STEPS == 0
        info = None
        if updated:
            if self.controller is None:
                self.target = self.anchor
                self.motors, inner = control(state, self.target)
                info = dict(mode='ORIGINAL_CONTROL', saturated=inner['saturated'])
                if self.hold_index is None:
                    self.hold_target, self.hold_index = self.anchor, self.index
            else:
                self.motors, info = self.controller.command(state)
                self.target = info['target']
                if self.controller.hold_target is not None and self.hold_index is None:
                    self.hold_target, self.hold_index = self.controller.hold_target, self.index
        self.index += 1
        return dict(updated=updated, target=None if self.target is None else list(self.target),
                    motors=list(self.motors), info=info)


def checkpoint(physics, stepper):
    return dict(index=stepper.index, time_s=physics.state()['time_s'],
                state_sha256=digest(integration_values(physics)), controller_sha256=digest(stepper.context()),
                clock=CLOCK, world=WORLD, model_sha256=physics.model_sha256,
                dll_sha256=physics.dll_sha256, implementation=stepper.implementation)


def summarize_trace(states, commands, stepper):
    times = [s['time_s']-states[0]['time_s'] for s in states]
    target, at = stepper.hold_target, stepper.hold_index
    flags = [at is not None and i >= at and hypot(*s['velocity']) <= .03
             and hypot(*(p-a for p, a in zip(s['position'], target))) <= .03
             and abs(s['position'][2]-target[2]) <= .1 for i, s in enumerate(states)]
    lower = [min(s['position'][j] for s in states) for j in range(3)]
    upper = [max(s['position'][j] for s in states) for j in range(3)]
    return dict(states=states, commands=commands,
        stop=dict(hold_target=None if target is None else list(target),
                  hold_latched_at_s=None if at is None else times[at], actual_hold_window=stable_window(times, flags)),
        envelope=dict(center_lower_m=lower, center_upper_m=upper, body_radius_m=.35, extra_margin_m=.02,
                      lower_m=[v-.37 for v in lower], upper_m=[v+.37 for v in upper],
                      horizon_end_s=states[-1]['time_s'], continuous_space_certified=False,
                      uncertainty_bound_validated=False),
        metrics=analyze(states, 0, stepper.recipe.direction), request_index=0,
        case=dict(direction=list(stepper.recipe.direction)), flight_authorized=False)


def predict_index(request):
    """Worker-only full rollout. Return a small monitor packet and separate full evidence."""
    record, _ = request_record(request)
    with QuadrotorPhysics() as physics:
        stepper = restore_request(physics, request)
        states, commands, points = [physics.state()], [], []
        for i in range(HORIZON_STEPS+1):
            if i % TICK_STEPS == 0:
                points.append(checkpoint(physics, stepper))
            if i == HORIZON_STEPS:
                break
            command = stepper.command(states[-1])
            physics.step(command['motors'])
            commands.append(command)
            states.append(physics.state())
        trace = summarize_trace(states, commands, stepper)
    # Suffix bounds include every 2 ms sample, including any later return motion.
    lower, upper = list(states[-1]['position']), list(states[-1]['position'])
    for i in range(HORIZON_STEPS, -1, -1):
        lower = [min(a, b) for a, b in zip(lower, states[i]['position'])]
        upper = [max(a, b) for a, b in zip(upper, states[i]['position'])]
        if i % TICK_STEPS == 0:
            points[i//TICK_STEPS]['remaining'] = dict(center_lower_m=lower, center_upper_m=upper,
                lower_m=[v-.37 for v in lower], upper_m=[v+.37 for v in upper],
                through_step=HORIZON_STEPS, continuous_space_certified=False, uncertainty_bound_validated=False)
    packet = FrozenForecast.create(dict(schema='async-stop-index-v1', request_sha256=request.sha256,
        request_id=record['request_id'], recipe=record['recipe'], checkpoints=points, execution_authorized=False))
    return packet, trace


class ForecastMonitor:
    """Consume one result; any timing/binding failure latches until a new research run."""
    def __init__(self, request):
        self.request = request
        self.record, self.recipe = request_record(request)
        self.packet = None
        self.points = None
        self.reason = 'WAITING'
        self.last_index = None
        self.last_elapsed = -1.
        self.accepted_at_step = None
        self.failed_at_step = None

    def _fail(self, reason, index):
        if self.reason in ('WAITING', 'MATCHED'):
            self.reason, self.failed_at_step = reason, index

    def _receive(self, packet):
        if type(packet) is not FrozenForecast:
            raise ValueError('immutable packet required')
        value = packet.record()
        if (value['schema'] != 'async-stop-index-v1' or value['request_sha256'] != self.request.sha256
            or value['request_id'] != self.record['request_id'] or value['recipe'] != self.record['recipe']
            or value['execution_authorized'] is not False):
            raise ValueError('packet identity or recipe differs')
        points = value['checkpoints']
        if len(points) != HORIZON_STEPS//TICK_STEPS+1:
            raise ValueError('incomplete monitor trajectory')
        expected = self.record['binding']
        for n, point in enumerate(points):
            if (point['index'] != n*TICK_STEPS
                or type(point['time_s']) not in (int, float) or not isfinite(point['time_s'])
                or abs(point['time_s']-(self.record['entry']['time_s']+n*TICK_SECONDS)) > 1e-8):
                raise ValueError('invalid checkpoint clock')
            for key in ('clock', 'world', 'model_sha256', 'dll_sha256'):
                if point[key] != expected[key]:
                    raise ValueError('checkpoint runtime differs')
            if point['implementation'] != self.record['implementation']:
                raise ValueError('checkpoint implementation differs')
            for key in ('state_sha256', 'controller_sha256'):
                if (not isinstance(point[key], str) or len(point[key]) != 64
                    or any(c not in '0123456789abcdef' for c in point[key])):
                    raise ValueError('invalid state digest')
            bounds = point['remaining']
            if (bounds['through_step'] != HORIZON_STEPS or bounds['continuous_space_certified'] is not False
                or bounds['uncertainty_bound_validated'] is not False):
                raise ValueError('invalid envelope claims')
            for key in ('center_lower_m', 'center_upper_m', 'lower_m', 'upper_m'):
                if len(bounds[key]) != 3 or any(type(v) not in (int, float) or not isfinite(v) for v in bounds[key]):
                    raise ValueError('invalid envelope coordinates')
            for j in range(3):
                if (bounds['center_lower_m'][j] > bounds['center_upper_m'][j]
                    or bounds['lower_m'][j] != bounds['center_lower_m'][j]-.37
                    or bounds['upper_m'][j] != bounds['center_upper_m'][j]+.37):
                    raise ValueError('invalid envelope expansion')
        self.packet, self.points = packet, points

    def tick(self, current, *, elapsed_s, packet=None, worker_error=False, deadline_missed=False):
        index = current['index']
        if type(index) is not int or index < 0 or index % TICK_STEPS:
            raise ValueError('integer monitor checkpoint required')
        if type(elapsed_s) not in (int, float) or not isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError('finite wall time required')
        if self.last_index is not None and index != self.last_index+TICK_STEPS:
            self._fail('MONITOR_SEQUENCE_ERROR', index)
        if elapsed_s < self.last_elapsed:
            self._fail('WALL_CLOCK_REVERSED', index)
        self.last_index, self.last_elapsed = index, elapsed_s
        if deadline_missed:
            self._fail('CONTROL_DEADLINE_MISSED', index)
        if worker_error:
            self._fail('WORKER_FAILED', index)
        if self.packet is None and elapsed_s > RESULT_DEADLINE:
            self._fail('RESULT_TIMEOUT', index)
        if self.reason in ('WAITING', 'MATCHED') and packet is not None:
            if self.packet is not None:
                self._fail('REPEATED_RESULT', index)
            else:
                try:
                    self._receive(packet)
                except (ValueError, KeyError, TypeError, IndexError):
                    self._fail('RESULT_BINDING_INVALID', index)
        remaining = None
        if self.reason in ('WAITING', 'MATCHED') and self.points is not None:
            if index >= HORIZON_STEPS:
                self._fail('HORIZON_EXHAUSTED', index)
            else:
                expected = self.points[index//TICK_STEPS]
                if any(current.get(key) != value for key, value in expected.items() if key != 'remaining'):
                    self._fail('EXECUTION_STATE_DIVERGED', index)
                else:
                    self.reason = 'MATCHED'
                    if self.accepted_at_step is None:
                        self.accepted_at_step = index
                    remaining = expected['remaining']
        return dict(reason=self.reason, forecast_matches=self.reason == 'MATCHED',
                    execution_authorized=False, accepted_at_step=self.accepted_at_step,
                    failed_at_step=self.failed_at_step, remaining=remaining)
