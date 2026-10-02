"""来源：本项目原创。预分配停止反馈记录；详细命令说明在循环外重建。"""
from array import array
from math import isfinite

from .async_stop import StopStepper, DT
from .stopping_policy import validate_state


class CompactStopLog:
    """Store actual states and actual motor commands without retaining nested dicts.

    This is a lossless record of scalar measurements, not a downsampling or a
    prediction. Command annotations reconstructed later are explicitly separate.
    """
    def __init__(self, initial, *, capacity=4000):
        if type(capacity) is not int or not 1 <= capacity <= 4000:
            raise ValueError('bounded positive step capacity required')
        validate_state(initial)
        self.capacity = capacity
        self.count = 0
        self.start_s = initial['time_s']
        self.states = array('d', [0.]) * ((capacity+1)*14)
        self.controls = array('d', [0.]) * (capacity*4)
        self._write_state(0, initial)

    def _write_state(self, index, state):
        base = index*14
        self.states[base] = state['time_s']
        offset = base+1
        for key in ('position', 'quaternion', 'velocity', 'angular_velocity'):
            for value in state[key]:
                self.states[offset] = value
                offset += 1

    def append(self, motors, state):
        if self.count >= self.capacity:
            raise ValueError('flight log capacity exhausted; no overwrite')
        validate_state(state)
        if abs(state['time_s']-(self.start_s+(self.count+1)*DT)) > 1e-8:
            raise ValueError('state sequence or model time differs')
        if len(motors) != 4 or any(type(v) not in (int,float) or not isfinite(v) or not 0 <= v <= 8 for v in motors):
            raise ValueError('four actual finite bounded motor commands required')
        # Validate everything before mutating the write position.
        for axis in range(4):
            self.controls[self.count*4+axis] = motors[axis]
        self._write_state(self.count+1, state)
        self.count += 1

    def state(self, index):
        if type(index) is not int or not 0 <= index <= self.count:
            raise IndexError('state has not been recorded')
        start = index*14
        return dict(time_s=self.states[start], position=list(self.states[start+1:start+4]),
                    quaternion=list(self.states[start+4:start+8]), velocity=list(self.states[start+8:start+11]),
                    angular_velocity=list(self.states[start+11:start+14]))

    def motors(self, index):
        if type(index) is not int or not 0 <= index < self.count:
            raise IndexError('motor command has not been recorded')
        return list(self.controls[index*4:index*4+4])

    def expand(self, record, recipe, *, transform_command):
        """Outside the timed loop: rebuild annotations and verify against observations."""
        if self.count != self.capacity:
            raise ValueError('incomplete log cannot be presented as a complete run')
        states = [self.state(i) for i in range(self.count+1)]
        if states[0] != record['entry']:
            raise ValueError('recorded entry differs from request')
        replay = StopStepper(record['entry'], recipe, record['prior_motors'])
        commands = []
        for index, state in enumerate(states[:-1]):
            command = transform_command(replay.command(state), index)
            observed = self.motors(index)
            if command['motors'] != observed:
                raise ValueError('reconstructed controller differs from recorded motors')
            command['motors'] = observed
            commands.append(command)
        return states, commands, replay
