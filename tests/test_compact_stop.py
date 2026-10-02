"""来源：本项目原创。无损紧凑记录与消费结束复查的异常边界。"""
from copy import deepcopy
import unittest

from drone_nav.async_stop import StopStepper, request_record
from drone_nav.compact_stop_log import CompactStopLog
from drone_nav.finished_space import consume_finished_space
from drone_nav.stop_forecast import CLOCK
from tools.async_stop_experiment import fault_command
import test_following_space as space_fixture
import test_async_stop as stop_fixture


class CompactStopTests(unittest.TestCase):
    def trace(self, steps=40):
        request, _, _ = stop_fixture.AsyncStopTests().fixture()
        record, recipe = request_record(request)
        stepper = StopStepper(record['entry'], recipe, record['prior_motors'])
        state = deepcopy(record['entry'])
        log = CompactStopLog(state, capacity=steps)
        states, commands = [deepcopy(state)], []
        for index in range(steps):
            command = fault_command(stepper.command(state), index, 'normal')
            state = dict(state, time_s=record['entry']['time_s']+(index+1)*.002)
            log.append(command['motors'], state)
            states.append(deepcopy(state)); commands.append(command)
        return log, record, recipe, stepper, states, commands

    def test_lossless_state_and_motor_roundtrip_and_controller_context(self):
        log, record, recipe, stepper, states, commands = self.trace()
        expanded, rebuilt, replay = log.expand(record,recipe,transform_command=lambda c,i:fault_command(c,i,'normal'))
        self.assertEqual(expanded,states); self.assertEqual(rebuilt,commands)
        self.assertEqual(replay.context(),stepper.context())
        self.assertEqual(log.states.buffer_info()[1],41*14)
        self.assertEqual(log.controls.buffer_info()[1],40*4)

    def test_arrays_do_not_alias_mutable_state_or_controls(self):
        log, _, _, _, states, commands = self.trace(2)
        commands[0]['motors'][0]=8.
        states[-1]['position'][0]=99.
        self.assertNotEqual(log.motors(0),commands[0]['motors'])
        self.assertNotEqual(log.state(2),states[-1])
        copy=log.state(0); copy['position'][0]=99.
        self.assertNotEqual(copy,log.state(0))

    def test_capacity_and_unwritten_entries_reject_without_overwrite(self):
        log,record,recipe,_,_,_=self.trace(1)
        previous=log.state(1)
        with self.assertRaises(ValueError): log.append([1.]*4,previous)
        self.assertEqual(log.state(1),previous)
        with self.assertRaises(IndexError): log.state(2)
        with self.assertRaises(IndexError): log.motors(1)
        for capacity in (0,4001,True):
            with self.assertRaises(ValueError): CompactStopLog(record['entry'],capacity=capacity)
        partial=CompactStopLog(record['entry'],capacity=2)
        with self.assertRaises(ValueError): partial.expand(record,recipe,transform_command=lambda c,i:c)

    def test_invalid_append_is_atomic(self):
        _,record,_,_,_,_=self.trace(1)
        log=CompactStopLog(record['entry'],capacity=2)
        valid=dict(record['entry'],time_s=7.002)
        for motors,state in (([float('nan')]*4,valid),([9.]*4,valid),([1.]*4,record['entry'])):
            with self.assertRaises(ValueError): log.append(motors,state)
            self.assertEqual(log.count,0)
            self.assertEqual(list(log.controls),[0.]*8)

    def test_reconstruction_cannot_replace_recorded_motor_truth(self):
        log,record,recipe,_,_,_=self.trace(2)
        log.controls[0]=8.
        with self.assertRaises(ValueError):
            log.expand(record,recipe,transform_command=lambda c,i:fault_command(c,i,'normal'))


class FinishedSpaceTests(unittest.TestCase):
    def check(self, times=(10.1,10.101), deadline=10.12, reader=None, **overrides):
        m,c,i,r,q=space_fixture.FollowingSpaceTests().fixture()
        m.tick(c(110),elapsed_s=.22)
        clock=iter(times)
        args=dict(capture_wall_s=10.,cycle_deadline_wall_s=deadline,clock_id=CLOCK,
                  clock_reader=lambda:next(clock),checkpoint_reader=lambda:c(110))
        if reader is not None: args['checkpoint_reader']=lambda:reader(m,c)
        args.update(overrides)
        return consume_finished_space(q,m,c(110),r,**args)

    def test_current_final_result_retains_all_original_limits(self):
        result=self.check()
        self.assertTrue(result['diagnostic_current'])
        self.assertFalse(result['flight_authorized'])
        self.assertIn('STOP_MODEL_UNCERTAINTY_UNVALIDATED',result['reasons'])
        self.assertGreater(result['finished_wall_s'],result['started_wall_s'])

    def test_crossing_control_deadline_during_consumption_rejects(self):
        result=self.check(times=(10.1,10.13))
        self.assertTrue(result['provisional']['diagnostic_current'])
        self.assertFalse(result['diagnostic_current'])
        self.assertIn('CONSUMPTION_FINISHED_AFTER_CONTROL_DEADLINE',result['final_failures'])

    def test_depth_expiry_during_consumption_rejects(self):
        result=self.check(times=(10.49,10.501),deadline=11.)
        self.assertTrue(result['provisional']['diagnostic_current'])
        self.assertFalse(result['diagnostic_current'])
        self.assertIn('DEPTH_EXPIRED_DURING_CONSUMPTION',result['final_failures'])

    def test_state_and_monitor_changes_cannot_reuse_provisional(self):
        result=self.check(reader=lambda m,c:c(120))
        self.assertIn('STATE_OR_MONITOR_CHANGED_DURING_CONSUMPTION',result['final_failures'])
        def fail(m,c):
            m.tick(c(120),elapsed_s=.24,deadline_missed=True)
            return c(110)
        self.assertFalse(self.check(reader=fail)['diagnostic_current'])

    def test_invalid_clock_values_and_reversed_time_reject(self):
        for times in ((10.1,float('nan')),(9.,10.1)):
            with self.assertRaises(ValueError): self.check(times=times)
        result=self.check(times=(10.1,10.09))
        self.assertIn('WALL_CLOCK_REVERSED_DURING_CONSUMPTION',result['final_failures'])

    def test_foreign_clock_does_not_produce_age_comparison(self):
        result=self.check(times=(10.1,1000.),wall_clock_id='foreign')
        self.assertFalse(result['diagnostic_current'])
        self.assertIn('WALL_CLOCK_MISMATCH',result['reasons'])
        self.assertNotIn('DEPTH_EXPIRED_DURING_CONSUMPTION',result['final_failures'])
        self.assertNotIn('CONSUMPTION_FINISHED_AFTER_CONTROL_DEADLINE',result['final_failures'])
        result=self.check(times=(1.,1.01),wall_clock_id='foreign')
        self.assertFalse(result['diagnostic_current'])
        self.assertIsNone(result['final_wall_age_s'])

    def test_already_invalid_result_never_recovers_on_finish(self):
        result=self.check(deadline_missed=True)
        self.assertFalse(result['provisional']['diagnostic_current'])
        self.assertFalse(result['diagnostic_current'])


if __name__=='__main__': unittest.main()
