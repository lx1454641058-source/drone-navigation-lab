"""来源：本项目原创。主动观察的原期限、范围和失败回执约束。"""
from copy import deepcopy
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from drone_nav.active_observation import ActiveObservationMixin,ProbeConfig,aim_at_box
from drone_nav.physical_vehicle import FlightBudget,spatial_cells
from drone_nav.pinhole import Intrinsics,Pose


class Base:
    def __init__(self):
        self.s=dict(time_s=7.2,position=[4.5,8.5,3.5],velocity=[0.,0.,0.],quaternion=[1.,0.,0.,0.],angular_velocity=[0.,0.,0.])
        self.confirmed_cell=(3,8);self.motor_latched_off=False;self.budget=FlightBudget()
        self.sampling_history=SimpleNamespace(_entries=[],grid=SimpleNamespace(occupied={(9,9,3)}),stamps={1:0.})
        self.move_checks=[]
    def state(self):return deepcopy(self.s)
    def move(self,target):
        self.confirmed_cell=tuple(target)
        return dict(completed=True,reason='ARRIVED_AND_SLOW',start=dict(time_s=2.4),end=self.state(),duration_s=4.8,
                    max_speed_mps=.1,max_height_error_m=0.,max_tracking_error_m=0.,stop_confirmed=True,stable_duration_s=.3)
    def scan(self,tick,aim):return 'normal-scan'


class Fake(ActiveObservationMixin,Base):pass


def support(vehicle,stop=10.9,source_end=12.):
    vehicle.confirmed_cell=(4,8)
    vehicle.move_checks=[dict(receipt={'completed':True},final_guard=dict(allowed=True,
        required_cells=[list(c) for c in spatial_cells((3,8),(4,8),.77)],baseline={'latest_stop_s':stop},
        sampling=[{'patches':[{'source':{'valid_until_s':source_end}}]}]))]


class ActiveObservationTests(unittest.TestCase):
    def test_missing_support_or_insufficient_time_does_not_start_control(self):
        v=Fake();v.observe_nearfield();self.assertEqual(v.probes[-1]['status'],'NOT_AUTHORIZED')
        support(v,stop=10.7);v.observe_nearfield()
        self.assertEqual(v.probes[-1]['status'],'INSUFFICIENT_TIME')
        self.assertEqual(v.state()['time_s'],7.2)
        support(v,stop=20.,source_end=10.7);v.observe_nearfield()
        self.assertEqual(v.probes[-1]['status'],'INSUFFICIENT_TIME')

    def test_nonrectangular_support_rejected(self):
        v=Fake();support(v);v.move_checks[-1]['final_guard']['required_cells'].remove([3,8,3])
        with self.assertRaises(ValueError):v.observe_nearfield()

    def test_failed_probe_cannot_reuse_arrival_stop_confirmation(self):
        v=Fake();v._navigation_aim=(5,8)
        def fail():
            v.s['time_s']=8.;v.s['velocity']=[.2,0.,0.]
            v.probes.append(dict(status='OBSERVING'))
            raise ValueError('camera failure')
        with patch.object(v,'observe_nearfield',side_effect=fail):r=v.move((4,8))
        self.assertFalse(r['completed']);self.assertFalse(r['stop_confirmed'])
        self.assertEqual(r['speed_mps'],.2)
        self.assertEqual(v.confirmed_cell,(3,8))
        self.assertTrue(r['arrival_receipt']['completed'])
        self.assertEqual(v.probes[-1]['status'],'PROBE_PROTOCOL_HOLD')

    def test_final_goal_and_disabled_probe_keep_original_move(self):
        for config,aim in ((ProbeConfig(),(4,8)),(ProbeConfig(enabled=False),(5,8))):
            v=Fake(probe_config=config);v._navigation_aim=aim
            with patch.object(v,'observe_nearfield') as probe:r=v.move((4,8))
            probe.assert_not_called();self.assertTrue(r['completed'])

    def test_retirement_does_not_refresh_stamps_or_clear_occupied_cells(self):
        v=Fake();v.s['time_s']=13.
        a=SimpleNamespace(captured_at_s=0.);b=SimpleNamespace(captured_at_s=1.)
        v.sampling_history._entries=[(a,'expired'),(b,'boundary')]
        v.retire_expired_views()
        self.assertEqual(v.sampling_history._entries,[(b,'boundary')])
        self.assertEqual(v.sampling_history.stamps,{1:0.})
        self.assertEqual(v.sampling_history.grid.occupied,{(9,9,3)})

    def test_projection_aim_fits_near_box_corners(self):
        s=dict(position=[4.52379,8.52961,3.5],quaternion=[1.,0.,0.,0.]);k=Intrinsics(80,60,12,12,39.5,29.5)
        low=(4.25,8.5,3.25);high=(4.5,8.75,3.5)
        target=aim_at_box(s,low,high,k);pose=Pose.look_at(tuple(s['position']),target)
        for corner in product(*zip(low,high)):
            p=pose.project(corner,k);self.assertIsNotNone(p)
            self.assertTrue(-.5<=p[0]<=79.5 and -.5<=p[1]<=59.5)

    def test_configuration_rejects_unbounded_or_ambiguous_values(self):
        for kw in (dict(offset_m=.13),dict(return_s=3.),dict(frame_interval_s=.01),dict(enabled=1),dict(offset_m=float('nan'))):
            with self.assertRaises(ValueError):ProbeConfig(**kw)


if __name__=='__main__':unittest.main()
