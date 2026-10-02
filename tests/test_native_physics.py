"""来源：本项目原创。实际物理引擎、解析对照、推力分配和资源生命周期检查。"""

import math
import unittest

from drone_nav.flight_control import ARM,YAW_RATIO,allocate,control,euler_quaternion
from drone_nav.native_physics import DEFAULT_DLL,QuadrotorPhysics
from drone_nav.physics_experiment import physics_cases,run_case


class AllocationTests(unittest.TestCase):
    def test_allocation_reconstructs_force_and_torques(self):
        motors,saturated=allocate(12,(.2,-.3,.02))
        self.assertFalse(saturated)
        self.assertAlmostEqual(sum(motors),12)
        self.assertAlmostEqual(ARM*(motors[0]-motors[1]+motors[2]-motors[3]),.2)
        self.assertAlmostEqual(ARM*(-motors[0]-motors[1]+motors[2]+motors[3]),-.3)
        self.assertAlmostEqual(YAW_RATIO*(motors[0]-motors[1]-motors[2]+motors[3]),.02)

    def test_saturation_is_reported(self):
        motors,saturated=allocate(80,(1,0,0))
        self.assertTrue(saturated)
        self.assertTrue(all(0<=v<=8 for v in motors))

    def test_bad_target_rejected(self):
        for target in ((1,2),(1,2,math.nan),(1,2,True)):
            with self.assertRaises(ValueError):control({},target)


@unittest.skipUnless(DEFAULT_DLL.exists(),'optional verified MuJoCo runtime is not installed')
class NativePhysicsTests(unittest.TestCase):
    def test_freefall_matches_independent_formula(self):
        with QuadrotorPhysics() as q:
            for _ in range(100):q.step([0]*4)
            state=q.state()
            self.assertAlmostEqual(state['position'][2],1-.5*9.81*.2**2,places=10)
            self.assertAlmostEqual(state['velocity'][2],-9.81*.2,places=10)

    def test_equal_rotor_thrust_holds(self):
        with QuadrotorPhysics() as q:
            for _ in range(500):q.step([1.4*9.81/4]*4)
            self.assertAlmostEqual(q.state()['position'][2],1,places=10)

    def test_differential_rotor_thrust_produces_roll(self):
        with QuadrotorPhysics() as q:
            base=1.4*9.81/4
            for _ in range(25):q.step([base+.1,base-.1,base+.1,base-.1])
            s=q.state()
            self.assertGreater(s['angular_velocity'][0],0)
            self.assertGreater(s['quaternion'][1],0)
            self.assertAlmostEqual(s['angular_velocity'][1],0,places=8)

    def test_closed_instance_rejects_use(self):
        q=QuadrotorPhysics();q.close();q.close()
        for action in (q.state,q.reset,q.ground_distance,lambda:q.step([0]*4)):
            with self.assertRaises(RuntimeError):action()

    def test_invalid_controls_do_not_advance_time(self):
        with QuadrotorPhysics() as q:
            for value in ([0]*3,[-1]*4,[9]*4,[math.nan]*4,[True]*4):
                with self.assertRaises(ValueError):q.step(value)
            self.assertEqual(q.state()['time_s'],0)

    def test_invalid_pose_rejected_without_reset(self):
        with QuadrotorPhysics() as q:
            q.step([0]*4);before=q.state()
            with self.assertRaises(ValueError):q.reset(quaternion=(2,0,0,0))
            self.assertEqual(q.state(),before)

    def test_invalid_xml_reports_error(self):
        with self.assertRaises(ValueError):QuadrotorPhysics(model_xml=b'<not_mujoco/>')

    def test_seven_bench_cases_and_timestamp_continuity(self):
        for case in physics_cases():
            with self.subTest(key=case['key']):
                result=run_case(case)
                self.assertTrue(result['check_passed'])
                self.assertEqual(len(result['commands']),round(case['duration_s']/.002))
                for i,s in enumerate(result['trace']):self.assertAlmostEqual(s['time_s'],i*.02,places=8)
                if case['mode'] not in ('loss','descent'):self.assertGreater(result['metrics']['min_ground_distance_m'],.7)

    def test_open_loop_command_replay_reproduces_feedback_run(self):
        original=run_case(physics_cases()[3])
        with QuadrotorPhysics() as q:
            for motors in original['commands']:q.step(motors)
            for key in ('position','quaternion','velocity','angular_velocity'):
                for a,b in zip(q.state()[key],original['trace'][-1][key]):self.assertAlmostEqual(a,b,places=10)


if __name__=='__main__':unittest.main()
