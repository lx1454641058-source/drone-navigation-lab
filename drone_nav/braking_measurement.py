"""来源：本项目原创。固定四旋翼停止试验及有限采样指标，不授权飞行。"""
from dataclasses import dataclass, asdict
from math import hypot, isfinite, sqrt

DT = .002
SPEED_TOLERANCE = .03
POSITION_TOLERANCE = .03
STABLE_SECONDS = .30
HEIGHT_TOLERANCE = .10
REACTION_SECONDS = .2
IDEAL_DECELERATION = 1.
EXTRA_MARGIN = .05


@dataclass(frozen=True)
class BrakeCase:
    name: str
    speed: float = .2
    direction: tuple = (1., 0.)
    ramp_seconds: float = 6.
    delay_seconds: float = 0.
    control_seconds: float = DT

    def __post_init__(self):
        numbers = (self.speed, self.ramp_seconds, self.delay_seconds, self.control_seconds, *self.direction)
        if len(self.direction) != 2 or any(type(v) not in (int, float) or not isfinite(v) for v in numbers):
            raise ValueError('finite numeric configuration required')
        if not self.name or not 0 < self.speed <= .2 or not 0 < self.ramp_seconds <= 6:
            raise ValueError('case outside fixed low-speed experiment')
        if not 0 <= self.delay_seconds <= .2 or not DT <= self.control_seconds <= .02:
            raise ValueError('invalid timing')
        if abs(hypot(*self.direction)-1) > 1e-9:
            raise ValueError('direction must be normalized')
        for value in (self.ramp_seconds, self.delay_seconds, self.control_seconds):
            if abs(value/DT-round(value/DT)) > 1e-8:
                raise ValueError('timings must be multiples of the physics step')


CASES = (
    BrakeCase('x05', .05), BrakeCase('x10', .1), BrakeCase('x20'),
    BrakeCase('y20', direction=(0., 1.)),
    BrakeCase('diagonal20', direction=(1/sqrt(2), 1/sqrt(2))),
    BrakeCase('negative20', direction=(-1., 0.)),
    BrakeCase('early20', ramp_seconds=.4),
    BrakeCase('delayed20', delay_seconds=.2),
    BrakeCase('control20ms', control_seconds=.02),
)


def stable_window(times, flags):
    """Only adjacent observed states count; the caller validates the sample clock."""
    start = None
    first_start = first_confirmation = None
    left_after_confirmation = False
    for t, valid in zip(times, flags):
        if not valid:
            start = None
            if first_confirmation is not None:
                left_after_confirmation = True
        else:
            if start is None:
                start = t
            if first_confirmation is None and t-start >= STABLE_SECONDS-1e-9:
                first_start, first_confirmation = start, t
    return dict(first_window_start_s=first_start, first_confirmation_s=first_confirmation,
                left_after_confirmation=left_after_confirmation,
                final_continuous_start_s=start,
                final_window_confirmed=start is not None and times[-1]-start >= STABLE_SECONDS-1e-9)


def analyze(states, request_index, direction):
    if type(request_index) is not int or not 0 <= request_index < len(states)-1:
        raise ValueError('request needs a state and subsequent samples')
    if len(direction) != 2 or any(not isfinite(v) for v in direction) or abs(hypot(*direction)-1) > 1e-9:
        raise ValueError('unit direction required')
    previous = None
    for state in states:
        for name, length in (('position', 3), ('velocity', 3), ('quaternion', 4), ('angular_velocity', 3)):
            if len(state[name]) != length or any(type(v) not in (int, float) or not isfinite(v) for v in state[name]):
                raise ValueError('invalid recorded state')
        time = state['time_s']
        if type(time) not in (int, float) or not isfinite(time) or time < 0:
            raise ValueError('invalid clock')
        if previous is not None and abs(time-previous-DT) > 1e-8:
            raise ValueError('nonconsecutive sample clock')
        previous = time
    samples = states[request_index:]
    entry = samples[0]
    times = [s['time_s']-entry['time_s'] for s in samples]
    v = hypot(*entry['velocity'][:2])
    axis = tuple(x/v for x in entry['velocity'][:2]) if v > 1e-9 else direction
    offsets = [[p-a for p, a in zip(s['position'], entry['position'])] for s in samples]
    forward = [p[0]*axis[0]+p[1]*axis[1] for p in offsets]
    lateral = [abs(p[0]*axis[1]-p[1]*axis[0]) for p in offsets]
    speeds = [hypot(*s['velocity']) for s in samples]
    slow = [x <= SPEED_TOLERANCE for x in speeds]
    hold = [ok and hypot(*p) <= POSITION_TOLERANCE and abs(p[2]) <= HEIGHT_TOLERANCE
            for ok, p in zip(slow, offsets)]
    ideal_distance = v*REACTION_SECONDS+v*v/(2*IDEAL_DECELERATION)
    ideal_time = REACTION_SECONDS+v/IDEAL_DECELERATION
    deadline_index = next((i for i, t in enumerate(times) if t >= ideal_time-1e-9), None)
    return dict(entry_horizontal_speed_mps=v, entry_speed_mps=speeds[0],
                entry_already_slow=slow[0], evaluation_direction=list(axis),
                observed_seconds=times[-1], samples=len(samples),
                max_forward_m=max(forward), max_backward_m=max(0., -min(forward)),
                max_lateral_m=max(lateral), max_horizontal_radius_m=max(hypot(*p[:2]) for p in offsets),
                horizontal_path_m=sum(hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(offsets, offsets[1:])),
                max_height_change_m=max(abs(p[2]) for p in offsets),
                max_horizontal_speed_mps=max(hypot(*s['velocity'][:2]) for s in samples),
                final_offset_m=offsets[-1], final_speed_mps=speeds[-1],
                speed_window=stable_window(times, slow), hold_window=stable_window(times, hold),
                ideal_distance_m=ideal_distance, ideal_stop_time_s=ideal_time,
                exceeds_ideal_distance=max(forward) > ideal_distance+1e-9,
                exceeds_ideal_distance_plus_margin=max(forward) > ideal_distance+EXTRA_MARGIN+1e-9,
                deadline_sample_s=None if deadline_index is None else times[deadline_index],
                speed_at_ideal_deadline_mps=None if deadline_index is None else speeds[deadline_index],
                slow_from_ideal_deadline=None if deadline_index is None else all(slow[deadline_index:]),
                flight_authorized=False, scope='sampled fixed-model response; not a physical bound')


def simulate(case):
    # Import the native library only in the experiment process, never in metric unit tests.
    from .native_physics import QuadrotorPhysics
    from .flight_control import control
    hover_steps, ramp_steps, tail_steps = round(1/DT), round(case.ramp_seconds/DT), round(8/DT)
    request = hover_steps+ramp_steps
    delay_steps, stride = round(case.delay_seconds/DT), round(case.control_seconds/DT)
    origin = (0., 0., 3.5)
    commands, states = [], []
    with QuadrotorPhysics() as physics:
        if physics.dt != DT:
            raise ValueError('physics timestep changed')
        physics.reset(origin)
        states.append(physics.state())
        motors = anchor = None
        for i in range(request+tail_steps):
            state = states[-1]
            if i == request:
                anchor = tuple(state['position'])
            if i < hover_steps:
                phase, target, phase_index = 'hover', origin, i
            elif i < request:
                phase, phase_index = 'ramp', i-hover_steps
                distance = case.speed*(phase_index*DT)
                target = (distance*case.direction[0], distance*case.direction[1], origin[2])
            elif i < request+delay_steps:
                phase, target, phase_index = 'delay_last_motors', None, i-request
            else:
                phase, target, phase_index = 'hold', anchor, i-request-delay_steps
            updated = phase != 'delay_last_motors' and phase_index % stride == 0
            saturated = None
            if updated:
                motors, info = control(state, target)
                saturated = info['saturated']
            commands.append(dict(phase=phase, target=target, updated=updated,
                                 motors=list(motors), saturated=saturated))
            physics.step(motors)
            states.append(physics.state())
        result = dict(case=asdict(case), initial_position=origin, request_index=request,
                      dt_s=DT, model_sha256=physics.model_sha256, dll_sha256=physics.dll_sha256,
                      engine_version=physics.version, states=states, commands=commands,
                      metrics=analyze(states, request, case.direction))
    return result
