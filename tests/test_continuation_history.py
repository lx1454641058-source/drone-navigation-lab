"""来源：本项目原创。长任务缓存、时间边界、原证据保存和失败原子性。"""
from copy import deepcopy
from dataclasses import replace
import unittest

from drone_nav.continuation_history import ContinuationHistory
from drone_nav.physical_vehicle import FlightBudget
from drone_nav.pinhole import Intrinsics,Pose
from drone_nav.raycast import World
from drone_nav.range_observation import render_range
from drone_nav.sampling_motion import SamplingView,SamplingAssumption


class ContinuationHistoryTests(unittest.TestCase):
    def view(self,index,t):
        k=Intrinsics(4,4,3,3,1.5,1.5)
        f=render_range(World(()),Pose.look_at((3.5,8.5,3.5),(4.5,8.5,3.5)),k,tick=index)
        return SamplingView(f,f'f-{index}','camera','physics-seconds',t,t)

    def test_long_capture_sequence_remains_bounded_without_extending_lifetime(self):
        h=ContinuationHistory();h.grid.occupied.add((5,8,3))
        for i in range(180):
            t=i*.04;h.add(self.view(i,t),now_s=t)
        self.assertLessEqual(h.peak_frames,89)
        self.assertEqual(len(h._ids),180)
        self.assertEqual(len(h.stamps),180)
        self.assertIn((5,8,3),h.grid.occupied)
        self.assertEqual(h.stamps[0],0.)
        self.assertTrue(h.retired)
        for entry in h.retired:
            self.assertEqual(entry['valid_until_s'],entry['captured_at_s']+12.)
            self.assertLess(entry['valid_until_s'],entry['earliest_next_stop_s'])

    def test_exact_future_stop_boundary_is_retained_then_retired(self):
        h=ContinuationHistory();h.add(self.view(0,0.),now_s=0.)
        h.add(self.view(1,3.5),now_s=3.5)
        self.assertEqual([v.frame_id for v,_ in h._entries],['f-0','f-1'])
        h.add(self.view(2,3.502),now_s=3.502)
        self.assertEqual([v.frame_id for v,_ in h._entries],['f-1','f-2'])
        self.assertEqual(h.retired[0]['frame_id'],'f-0')

    def test_invalid_arrival_does_not_remove_old_frames_or_commit_retirement(self):
        h=ContinuationHistory();h.add(self.view(0,0.),now_s=0.)
        original_entries=h._entries
        before=deepcopy(h.__dict__)
        for bad in (replace(self.view(1,4.),clock_id='wrong'),
                    replace(self.view(1,4.),available_at_s=5.)):
            with self.assertRaises(ValueError):h.add(bad,now_s=4.)
            self.assertIs(h._entries,original_entries)
            self.assertEqual([v for v,_ in h._entries],[v for v,_ in before['_entries']])
            self.assertEqual(h.grid.__dict__,before['grid'].__dict__)
            self.assertEqual(h.retired,before['retired'])
            self.assertEqual(h.stamps,before['stamps'])
            self.assertEqual(h._now,before['_now'])

    def test_removed_source_identity_cannot_be_reused(self):
        h=ContinuationHistory();h.add(self.view(0,0.),now_s=0.)
        h.add(self.view(1,4.),now_s=4.)
        with self.assertRaises(ValueError):
            h.add(replace(self.view(2,5.),frame_id='f-0'),now_s=5.)

    def test_budget_change_or_backward_clock_rejected_before_mutation(self):
        h=ContinuationHistory();h.add(self.view(0,1.),now_s=1.)
        with self.assertRaises(ValueError):h.add(self.view(1,.5),now_s=.5)
        for b in (replace(FlightBudget(),max_move_s=7.),replace(FlightBudget(),free_ttl_s=13.)):
            with self.assertRaises(ValueError):
                h.check((3,8),(4,8),now_s=1.,budget=b,assumption=SamplingAssumption(.2,.2))
        self.assertEqual(h._now,1.)

    def test_invalid_fixed_contract_rejected(self):
        for kw in (dict(horizon_s=True),dict(horizon_s=12.),dict(free_ttl_s=float('nan')),dict(horizon_s=0.)):
            with self.assertRaises(ValueError):ContinuationHistory(**kw)


if __name__=='__main__':unittest.main()
