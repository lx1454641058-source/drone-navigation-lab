"""来源：本项目原创。姿态/角速度反馈减速，仅用于固定机体仿真研究。"""
from math import hypot, isfinite
from .flight_control import control, GRAVITY

KV, KA, KJ = 6.75, 2.375, .2875
MAX_ACCELERATION = 2.5
MAX_SAMPLE_GAP = .021
QUIET_SECONDS = .3


def validate_state(state):
    for field, length in (('position', 3), ('velocity', 3), ('quaternion', 4), ('angular_velocity', 3)):
        values = state[field]
        if len(values) != length or any(type(v) not in (int, float) or not isfinite(v) for v in values):
            raise ValueError('invalid physical feedback')
    time = state['time_s']
    if type(time) not in (int, float) or not isfinite(time) or time < 0:
        raise ValueError('invalid physical clock')
    if abs(hypot(*state['quaternion'])-1) > 1e-6:
        raise ValueError('unit quaternion required')


def tilt_response(state):
    """World-frame thrust-axis slope and its rate; body-frame angular velocity."""
    validate_state(state)
    w, x, y, z = state['quaternion']
    ox, oy, oz = state['angular_velocity']
    rotation = ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
                (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
                (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))
    up = [row[2] for row in rotation]
    if up[2] < .7:
        raise ValueError('tilt outside experiment envelope')
    # omega x body_up = (omega_y, -omega_x, 0); rotate once into world coordinates.
    rate = [row[0]*oy-row[1]*ox for row in rotation]
    acceleration = [GRAVITY*up[i]/up[2] for i in range(2)]
    jerk = [GRAVITY*(rate[i]*up[2]-up[i]*rate[2])/(up[2]*up[2]) for i in range(2)]
    return acceleration, jerk


class AttitudeStop:
    """First attenuate motion, then latch the actual position. Never resumes a route."""
    def __init__(self, request_state):
        validate_state(request_state)
        tilt_response(request_state)
        self.request_time = request_state['time_s']
        self.height = request_state['position'][2]
        self.last_time = None
        self.quiet_since = None
        self.hold_target = None
        self.hold_at = None
        self.fault = None

    def command(self, state):
        if self.fault:
            raise ValueError('stopping controller is fault-latched: '+self.fault)
        try:
            acceleration, jerk = tilt_response(state)
            now = state['time_s']
            if now < self.request_time-1e-9 or (self.last_time is not None and now <= self.last_time):
                raise ValueError('repeated or reversed feedback clock')
            quiet_since = self.quiet_since
            if self.last_time is not None and now-self.last_time > MAX_SAMPLE_GAP+1e-9:
                quiet_since = None
            quiet = (hypot(*state['velocity']) <= .01 and hypot(*acceleration) <= .05
                     and hypot(*state['angular_velocity']) <= .05
                     and abs(state['position'][2]-self.height) <= .10)
            if not quiet:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            target, hold_at = self.hold_target, self.hold_at
            if target is None and quiet_since is not None and now-quiet_since >= QUIET_SECONDS-1e-9:
                target, hold_at = tuple(state['position']), now
            raw = [-KV*state['velocity'][i]-KA*acceleration[i]-KJ*jerk[i] for i in range(2)]
            desired = [max(-MAX_ACCELERATION, min(MAX_ACCELERATION, a)) for a in raw]
            if target is None:
                virtual = [state['position'][i]+(desired[i]+2.8*state['velocity'][i])/1.8 for i in range(2)]
                virtual.append(self.height)
            else:
                virtual = target
            motors, inner = control(state, virtual)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self.fault = str(exc)
            raise ValueError('invalid stop feedback: '+str(exc)) from exc
        # Commit timing/latch state only after a complete valid command was formed.
        self.last_time, self.quiet_since = now, quiet_since
        self.hold_target, self.hold_at = target, hold_at
        info = dict(mode='POSITION_HOLD' if target is not None else 'ATTITUDE_BRAKE',
                    target=list(virtual), target_is_navigation_waypoint=False,
                    estimated_acceleration_mps2=acceleration, estimated_jerk_mps3=jerk,
                    raw_acceleration_mps2=raw, requested_acceleration_mps2=desired,
                    acceleration_clipped=raw != desired, saturated=inner['saturated'],
                    quiet=quiet, quiet_since_s=quiet_since, hold_at_s=hold_at,
                    hold_target=None if target is None else list(target), flight_authorized=False)
        return motors, info
