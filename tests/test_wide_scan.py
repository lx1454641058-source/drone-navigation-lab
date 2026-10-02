"""来源：本项目原创。扫描标定、采集时间与序列，以及路段故障触发。"""
from types import SimpleNamespace
from dataclasses import replace
import unittest
from unittest.mock import patch
from drone_nav.wide_scan import WideScanMixin
from drone_nav.mission_goal_probe import MissionGoalProbeMixin
from drone_nav.pinhole import PerspectiveFrame,Pose,Intrinsics
from tools.building_route_experiment import Vehicle,base_spec


class History:
    def __init__(self):self.grid=SimpleNamespace(last_tick=-1);self.views=[]
    def add(self,v,*,now_s):
        v.validate()
        assert v.frame.tick==self.grid.last_tick+1 and now_s==v.available_at_s
        self.views.append(v);self.grid.last_tick=v.frame.tick


class Base:
    def __init__(self):
        self.spec={'wide_scan':True};self.probes=[];self.now=0.
        self.sampling_history=History();self.source_frames=[];self.capture_times=[]
    def state(self):return {'time_s':self.now,'position':(3.5,8.5,3.5)}
    def hold(self,s):self.now+=s
    def retire_expired_views(self):pass
    def scan(self,tick,aim):return 'original'
    def capture(self,k,tick,aim):
        self.capture_times.append(self.now)
        return PerspectiveFrame(k,aim((3.5,8.5,3.5)),((0,0,0),)*(k.width*k.height),(None,)*(k.width*k.height),tick)


class Fake(WideScanMixin,Base):pass


class ProbeBase:
    def move(self,target):
        if getattr(self,'fail',False):raise ValueError('failed move')
        return dict(will_probe=tuple(target)!=self._navigation_aim)


class ProbeFake(MissionGoalProbeMixin,ProbeBase):pass


class WideScanTests(unittest.TestCase):
    def test_full_initial_scan_preserves_source_times_and_calibration(self):
        v=Fake();frames=v.scan(7,(9,8))
        self.assertEqual(len(frames),16);self.assertAlmostEqual(v.now,1.8)
        self.assertTrue(all(f.intrinsics==Intrinsics(80,60,25,25,39.5,29.5) and f.tick==7 for f in frames))
        self.assertEqual([a.captured_at_s for a in v.sampling_history.views],v.capture_times)
        self.assertEqual([a.frame.tick for a in v.sampling_history.views],list(range(16)))
        self.assertTrue(all(abs(a.available_at_s-.9)<1e-9 for a in v.sampling_history.views[:8]))
        self.assertTrue(all(abs(a.available_at_s-1.8)<1e-9 for a in v.sampling_history.views[8:]))

    def test_after_confirmed_probe_only_eight_fresh_navigation_views(self):
        v=Fake();v.probes=[{'status':'RETURN_CONFIRMED'}]
        self.assertEqual(len(v.scan(1,(9,8))),8);self.assertAlmostEqual(v.now,.9)
        self.assertTrue(all(a.frame_id.startswith('nav-') for a in v.sampling_history.views))

    def test_disabled_uses_original_scan(self):
        v=Fake();v.spec['wide_scan']=False
        self.assertEqual(v.scan(0,(9,8)),'original');self.assertEqual(v.now,0.)

    def test_level_ring_observes_fixed_directions_without_extra_time(self):
        v=Fake();v.spec['level_scan']=True;v.probes=[{'status':'RETURN_CONFIRMED'}]
        frames=v.scan(1,(3,7))
        self.assertEqual(len(frames),8);self.assertAlmostEqual(v.now,.9)
        self.assertTrue(all(abs(f.pose.forward[2])<1e-12 for f in frames))
        self.assertEqual(frames[0].pose.forward,(1.,0.,0.))

    def test_bad_calibration_is_rejected_before_history_update(self):
        v=Fake();old=v.capture
        def wrong(k,tick,aim):return replace(old(k,tick,aim),intrinsics=replace(k,fx=26))
        v.capture=wrong
        with self.assertRaises(ValueError):v.scan(0,(9,8))
        self.assertFalse(v.sampling_history.views)

    def test_invalid_scenario_options_rejected_before_native_allocation(self):
        for changes in ({'sensor_after_moves':0},{'sensor_after_moves':True},{'ignore_move':-1},{'wide_scan':1},{'level_scan':True}):
            with self.assertRaises(ValueError):Vehicle(dict(base_spec(),**changes))

    def test_sensor_stage_is_replayed_and_original_spec_restored(self):
        v=object.__new__(Vehicle);v.spec=dict(base_spec(),sensor_after_moves=2)
        v.saved=None;v.frames=[];v.fault_frames=0;v.actions=[{},{}]
        original=v.spec
        def capture(target,*args):
            self.assertEqual(target.spec['range_fault'],'all')
            target.frames.append({});return 'frame'
        with patch('tools.building_route_experiment.RangeVehicle.capture',side_effect=capture):
            self.assertEqual(v.capture(None,0),'frame')
        self.assertIs(v.spec,original);self.assertTrue(v.frames[-1]['sensor_fault_active'])
        self.assertEqual(v.fault_frames,1)
        v.saved={};v.saved_frames=[{'sensor_fault_active':False}];v.frame_index=0
        with self.assertRaises(ValueError):v.capture(None,0)

    def test_frontier_does_not_skip_probe_but_real_goal_does(self):
        v=ProbeFake();v.spec={'goal':[9,8],'mission_goal_probe':True};v._navigation_aim=(5,5)
        self.assertTrue(v.move((5,5))['will_probe'])
        self.assertEqual(v._navigation_aim,(5,5))
        self.assertFalse(v.move((9,8))['will_probe'])
        v.spec['mission_goal_probe']=False
        self.assertFalse(v.move((5,5))['will_probe'])

    def test_move_failure_restores_scan_aim(self):
        v=ProbeFake();v.spec={'goal':[9,8],'mission_goal_probe':True};v._navigation_aim=(5,5);v.fail=True
        with self.assertRaises(ValueError):v.move((5,5))
        self.assertEqual(v._navigation_aim,(5,5))


if __name__=='__main__':unittest.main()
