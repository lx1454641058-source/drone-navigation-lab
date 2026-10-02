"""来源：本项目原创。任务终态、停桨锁、异常恢复与原流程集成回归。"""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from drone_nav.classifier import ColorModel
from drone_nav.mission_supervisor import MissionSupervisor, MissionVehicle, MotorInterlockError
from drone_nav.native_physics import DEFAULT_DLL
from drone_nav.raycast import World, Surface
from tools.supervised_mission_experiment import MODEL, specs, simulate_case


class FakeVehicle:
    def __init__(self):
        self.motor_latched_off = False
        self.disarm_requested_at_s = None
        self.operation_phase = 'CREATED'
        self.current = dict(time_s=0., position=[3.5,8.5,3.5], velocity=[0.,0.,0.],
                            quaternion=[1.,0.,0.,0.], angular_velocity=[0.,0.,0.])

    def state(self):
        return deepcopy(self.current)


def nav(terminal='READY_TO_LAND'):
    return dict(terminal_state=terminal, trace=[dict(position=[4,8])])


class SupervisorContractTests(unittest.TestCase):
    def make(self, v=None):
        return MissionSupervisor(v or FakeVehicle(), SimpleNamespace(digest='test'), (3,8), (4,8))

    def test_preexisting_disarm_never_enters_navigation(self):
        v=FakeVehicle();v.motor_latched_off=True
        with patch('drone_nav.mission_supervisor.navigate_physical') as navigate:
            r=self.make(v).run()
        self.assertEqual(r['state'],'DISARMED_UNCONFIRMED');navigate.assert_not_called()

    def test_invalid_initial_state_does_not_start_or_recover(self):
        for key,value in [('velocity',[float('nan'),0.,0.]),('position',[4.,8.5,3.5])]:
            v=FakeVehicle();v.current[key]=value
            with patch('drone_nav.mission_supervisor.navigate_physical') as navigate:
                r=self.make(v).run()
            self.assertEqual(r['state'],'MANUAL_REVIEW');navigate.assert_not_called()
            self.assertIsNone(r['recovery'])

    def test_navigation_rejection_skips_descent_and_preserves_cause(self):
        with patch('drone_nav.mission_supervisor.navigate_physical',return_value=nav('LANDING_REJECTED')), \
             patch('drone_nav.mission_supervisor.descend_after_navigation') as descend, \
             patch('drone_nav.mission_supervisor.recover_hover',return_value={'motion_stable':True,'state':'MOTION_STABLE'}):
            r=self.make().run()
        descend.assert_not_called()
        self.assertEqual(r['state'],'ABORTED_MOTION_STABLE')
        self.assertEqual(r['original_reason'],'NAVIGATION:LANDING_REJECTED')
        self.assertFalse(r['delivery_completed']);self.assertFalse(r['resume_allowed'])

    def test_disarmed_unconfirmed_never_calls_recovery(self):
        v=FakeVehicle()
        def touchdown(*args,**kwargs):
            v.motor_latched_off=True;v.disarm_requested_at_s=1.
            return dict(status='TOUCHDOWN_UNCONFIRMED',stop_confirmed=False,disarmed_at_s=1.)
        with patch('drone_nav.mission_supervisor.navigate_physical',return_value=nav()), \
             patch('drone_nav.mission_supervisor.descend_after_navigation',side_effect=touchdown), \
             patch('drone_nav.mission_supervisor.recover_hover') as recovery:
            r=self.make(v).run()
        recovery.assert_not_called();self.assertEqual(r['state'],'DISARMED_UNCONFIRMED')
        self.assertFalse(r['touchdown_confirmed'])

    def test_protocol_exception_after_disarm_cannot_restart_motors(self):
        v=FakeVehicle()
        def failure(*args,**kwargs):
            v.motor_latched_off=True
            raise ValueError('probe protocol failed')
        with patch('drone_nav.mission_supervisor.navigate_physical',return_value=nav()), \
             patch('drone_nav.mission_supervisor.descend_after_navigation',side_effect=failure), \
             patch('drone_nav.mission_supervisor.recover_hover') as recovery:
            r=self.make(v).run()
        recovery.assert_not_called();self.assertEqual(r['state'],'DISARMED_UNCONFIRMED')

    def test_armed_protocol_exception_enters_recovery_failure(self):
        with patch('drone_nav.mission_supervisor.navigate_physical',side_effect=ValueError('bad camera')), \
             patch('drone_nav.mission_supervisor.recover_hover',return_value={'motion_stable':False,'state':'RECOVERY_TIMEOUT'}):
            r=self.make().run()
        self.assertEqual(r['state'],'ABORTED_RECOVERY_UNCONFIRMED')
        self.assertIn('bad camera',r['original_reason'])

    def test_missing_report_field_is_a_protocol_abort(self):
        with patch('drone_nav.mission_supervisor.navigate_physical',return_value={}), \
             patch('drone_nav.mission_supervisor.recover_hover',return_value={'motion_stable':True,'state':'MOTION_STABLE'}):
            r=self.make().run()
        self.assertEqual(r['state'],'ABORTED_MOTION_STABLE')
        self.assertTrue(r['original_reason'].startswith('PROTOCOL:'))

    def test_disarm_during_recovery_overrides_stable_receipt(self):
        v=FakeVehicle()
        def recovery(*args,**kwargs):
            v.motor_latched_off=True
            return {'motion_stable':True,'state':'MOTION_STABLE'}
        with patch('drone_nav.mission_supervisor.navigate_physical',return_value=nav('SENSOR_HOLD')), \
             patch('drone_nav.mission_supervisor.recover_hover',side_effect=recovery):
            r=self.make(v).run()
        self.assertEqual(r['state'],'DISARMED_UNCONFIRMED')

    def test_terminal_cannot_resume_and_illegal_transition_rejected(self):
        v=FakeVehicle();v.motor_latched_off=True;s=self.make(v);s.run()
        with self.assertRaises(RuntimeError):s.run()
        with self.assertRaises(RuntimeError):s.transition('CRUISING','retry')

    def test_bad_configuration_rejected_before_run(self):
        for start in [(0,8),(3,True),(3,),None]:
            with self.assertRaises(ValueError):MissionSupervisor(FakeVehicle(),None,start,(4,8))
        with self.assertRaises(ValueError):MissionSupervisor(FakeVehicle(),None,(3,8),(4,8),recovery_budget=False)


@unittest.skipUnless(DEFAULT_DLL.exists(),'verified native physics runtime unavailable')
class NativeSupervisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=ColorModel.from_dict(json.loads(MODEL.read_text(encoding='utf-8')))
        wanted={'paved','water','short-abort','post-disarm','recovery-fault'}
        cls.results={s['key']:simulate_case(s,cls.model) for s in specs() if s['key'] in wanted}

    def test_success_is_touchdown_not_delivery(self):
        raw,s=self.results['paved'];r=s['result']
        self.assertEqual(r['state'],'LANDED_CONFIRMED');self.assertTrue(r['motors_off'])
        self.assertTrue(r['touchdown_confirmed']);self.assertFalse(r['delivery_completed'])
        self.assertIsNone(r['recovery'])
        self.assertEqual([e['state'] for e in r['events']],['CRUISING','LANDING_AUTHORIZED','DESCENDING','LANDED_CONFIRMED'])
        self.assertTrue(all(m==[0.,0.,0.,0.] for m in raw['commands'][-500:]))

    def test_rejected_navigation_and_short_descent_abort_recover(self):
        for key in ('water','short-abort'):
            _,s=self.results[key];r=s['result']
            self.assertEqual(r['state'],'ABORTED_MOTION_STABLE')
            self.assertTrue(r['recovery']['motion_stable'])
            self.assertFalse(r['motors_off'])
        self.assertIsNone(self.results['water'][1]['result']['descent'])
        self.assertEqual(self.results['short-abort'][1]['result']['descent']['status'],'ABORT_STOP_UNCONFIRMED')

    def test_post_disarm_failure_has_no_new_recovery_or_thrust(self):
        raw,s=self.results['post-disarm'];r=s['result']
        self.assertEqual(r['state'],'DISARMED_UNCONFIRMED')
        self.assertIsNone(r['recovery']);self.assertFalse(r['touchdown_confirmed'])
        first_zero=next(i for i,m in enumerate(raw['commands']) if all(v==0 for v in m))
        self.assertTrue(all(all(v==0 for v in m) for m in raw['commands'][first_zero:]))

    def test_persistent_recovery_fault_remains_unconfirmed(self):
        r=self.results['recovery-fault'][1]['result']
        self.assertEqual(r['state'],'ABORTED_RECOVERY_UNCONFIRMED')
        self.assertEqual(r['recovery']['state'],'RECOVERY_ENVELOPE_BREACH')

    def test_all_control_paths_obey_motor_interlock(self):
        v=MissionVehicle(World((Surface('ground',0,(-40,60,-40,56)),)),(3,8))
        try:
            v.apply_motors([0.]*4);count=len(v.commands);state=v.state()
            for command in (lambda:v.apply_motors([1.]*4),lambda:v._step((3.5,8.5,3.5)),
                            lambda:v.hold(.1),lambda:v.move((4,8))):
                with self.assertRaises(MotorInterlockError):command()
                self.assertEqual(len(v.commands),count);self.assertEqual(v.state(),state)
        finally:v.close()

    def test_saved_camera_and_contact_replay_supervisor(self):
        raw,expected=self.results['post-disarm']
        with patch('drone_nav.physical_vehicle.render',side_effect=AssertionError('must replay saved images')):
            replay,result=simulate_case(expected['spec'],self.model,raw)
        self.assertEqual(replay,raw);self.assertEqual(result,expected)


if __name__=='__main__':unittest.main()
