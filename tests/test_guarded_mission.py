"""来源：本项目原创。主任务移动门槛、时钟保留和回执推进检查。"""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from drone_nav.classifier import ColorModel
from drone_nav.descent_experiment import descent_cases
from drone_nav.guarded_mission import GuardedMissionVehicle, _CheckedExecutor
from drone_nav.mission_supervisor import MissionSupervisor, MotorInterlockError
from drone_nav.physical_vehicle import PhysicalVehicle
from drone_nav.sampling_motion import SamplingAssumption


class GuardedMissionTests(unittest.TestCase):
    def setUp(self):
        self.world = next(x['world'] for x in descent_cases() if x['key'] == 'paved')
        self.model = ColorModel.from_dict(json.loads(Path('work/run-v11-descent-validated/model.json').read_text()))
        self.v = GuardedMissionVehicle(self.world, (3,8), sampling_assumption=SamplingAssumption(.2,.2))
        self.addCleanup(self.v.close)

    def test_native_mission_refuses_uncovered_edge_and_skips_descent(self):
        result = MissionSupervisor(self.v,self.model,(3,8),(4,8)).run()
        self.assertEqual(result['state'],'ABORTED_MOTION_STABLE')
        self.assertEqual(result['confirmed_cell'],[3,8])
        self.assertIsNone(result['descent'])
        self.assertEqual(self.v.actions,[])
        check=self.v.move_checks[0]
        self.assertTrue(check['initial']['baseline']['allowed'])
        self.assertFalse(check['final_guard']['allowed'])
        self.assertEqual(len(check['events']),4)
        self.assertEqual(self.v.audit['clearance_violations'],0)

    def test_navigation_batch_and_sampling_sequence_preserve_capture_age(self):
        frames=self.v.scan(0,(4,8))
        self.assertEqual({f.tick for f in frames},{0})
        self.assertEqual([x['ledger_tick'] for x in self.v.source_frames],list(range(8)))
        for (view,_),source,raw in zip(self.v.sampling_history._entries,self.v.source_frames,self.v.frames):
            self.assertEqual(view.captured_at_s,raw['capture_state']['time_s'])
            self.assertEqual(source['navigation_tick'],raw['frame']['tick'])
            self.assertAlmostEqual(view.available_at_s,.9)
        self.v.move((4,8))
        self.v.scan(1,(4,8))
        self.assertEqual([x['ledger_tick'] for x in self.v.source_frames][-8:],list(range(12,20)))
        self.assertEqual(self.v.sampling_history._entries[0][0].captured_at_s,0.)

    def test_confirmed_cell_advances_only_after_completed_receipt(self):
        checks=[]
        def fake(executor, history, start, end, *args, **kwargs):
            checks.append((start,tuple(end)))
            return {'receipt':{'completed':len(checks)==1,'reason':'test'},'state':'test'}
        with patch('drone_nav.guarded_mission.rescan_and_move',side_effect=fake):
            self.v.move((4,8));self.v.move((5,8))
        self.assertEqual(checks,[((3,8),(4,8)),((4,8),(5,8))])
        self.assertEqual(self.v.confirmed_cell,(4,8))
        self.assertEqual(len(self.v.move_checks),2)

    def test_checked_executor_calls_existing_motor_controller_without_recursion(self):
        executor=_CheckedExecutor(self.v)
        with patch.object(PhysicalVehicle,'move',return_value={'completed':True}) as raw:
            executor.move((4,8))
        raw.assert_called_once_with(self.v,(4,8))
        executor.hold_target=(3.,8.,3.5)
        self.assertEqual(self.v.hold_target,(3.,8.,3.5))

    def test_interlock_rejects_before_any_scan_or_motor(self):
        self.v.apply_motors([0.,0.,0.,0.])
        before=self.v.state();steps=self.v.steps
        with self.assertRaises(MotorInterlockError):self.v.move((4,8))
        self.assertEqual(self.v.steps,steps)
        self.assertEqual(self.v.state(),before)
        self.assertEqual(self.v.move_checks,[])

    def test_rescan_exception_clears_phase_flag_and_supervisor_recovers(self):
        with patch('drone_nav.guarded_mission.rescan_and_move',side_effect=ValueError('bad scan')):
            r=MissionSupervisor(self.v,self.model,(3,8),(4,8)).run()
        self.assertFalse(self.v.rescanning)
        self.assertEqual(r['original_reason'],'PROTOCOL:bad scan')
        self.assertEqual(r['state'],'ABORTED_MOTION_STABLE')
        self.assertEqual(self.v.actions,[])


if __name__=='__main__':unittest.main()
