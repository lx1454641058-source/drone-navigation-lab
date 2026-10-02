"""来源：本项目原创。3D 停止体积和候选规划的模型边界测试。"""
from dataclasses import replace
import math
import unittest

from drone_nav.motion import MotionConfig,profile,simulate
from drone_nav.motion_volume import RestMotionCandidate,stopping_volume,evaluate_motion,planning_comparison
import test_depth_volume as depth_fixture


class MotionVolumeTests(unittest.TestCase):
    def inspector(self): return depth_fixture.DepthVolumeTests().bind(*depth_fixture.inputs())
    def candidate(self,start=(0.,0.,1.3),end=(0.,0.,2.3)):
        view=self.inspector()
        return RestMotionCandidate('forward',view.world_frame,view.reference_fingerprint,start,end)

    def test_known_distance_stop_time_and_relative_error(self):
        box,r=stopping_volume(self.candidate(),MotionConfig(),now_s=1.2,error_bound_m=.1)
        self.assertAlmostEqual(r['required_distance_m'],1.35)
        self.assertAlmostEqual(r['latest_stop_s'],3.4)
        self.assertAlmostEqual(r['relative_padding_m'],.45)
        self.assertAlmostEqual(box.lower[2],.85); self.assertAlmostEqual(box.upper[2],3.1)

    def test_axis_negative_and_diagonal_emergency_trajectories_are_enclosed(self):
        config=MotionConfig()
        for end in ((1.,0.,2.),(-1.,0.,2.),(1.,1.,3.),(0.,0.,1.)):
            candidate=self.candidate((0.,0.,2.),end)
            box,r=stopping_volume(candidate,config,now_s=2.,error_bound_m=.03)
            p=profile(math.dist(candidate.start,candidate.end),config)
            for i in range(21):
                trace=simulate(p,config,failure_at_s=p.total_s*i/21)
                self.assertLessEqual(trace['duration_s'],r['latest_stop_s']-2.+1e-9)
                for sample in trace['samples']:
                    center=tuple(a+d*sample['distance_m'] for a,d in zip(candidate.start,r['direction']))
                    for axis in range(3):
                        self.assertGreaterEqual(center[axis]-r['relative_padding_m'],box.lower[axis]-1e-9)
                        self.assertLessEqual(center[axis]+r['relative_padding_m'],box.upper[axis]+1e-9)

    def test_point_in_front_is_not_the_entire_starting_body(self):
        view=self.inspector(); candidate=self.candidate((0.,0.,0.),(0.,0.,.5))
        r=evaluate_motion(view,candidate,MotionConfig(),now_s=1.2,clock_id='clock',error_bound_m=0.)
        self.assertIn('VOLUME_NOT_FULLY_IN_FRONT',r['reasons'])
        self.assertFalse(r['flight_authorized'])

    def test_current_depth_expires_before_long_motion_stops(self):
        r=evaluate_motion(self.inspector(),self.candidate(),MotionConfig(),now_s=1.2,clock_id='clock',error_bound_m=0.)
        self.assertTrue(r['expires_before_stop'])
        self.assertIn('OBSERVATION_EXPIRES_BEFORE_STOP',r['reasons'])

    def test_short_timely_motion_still_requires_coverage_and_physical_bounds(self):
        r=evaluate_motion(self.inspector(),self.candidate(end=(0.,0.,1.31)),MotionConfig(reaction_s=0.),
            now_s=1.2,clock_id='clock',error_bound_m=0.)
        self.assertFalse(r['expires_before_stop'])
        self.assertIn('PIXEL_GAPS_AND_ERROR_BOUNDS_UNVERIFIED',r['reasons'])
        self.assertFalse(r['selected_for_execution'])

    def test_reference_and_clock_mismatch_rejected(self):
        view=self.inspector(); candidate=self.candidate()
        with self.assertRaises(ValueError): evaluate_motion(view,replace(candidate,reference_fingerprint='b'*64),MotionConfig(),now_s=1.2,clock_id='clock',error_bound_m=0.)
        with self.assertRaises(ValueError): evaluate_motion(view,candidate,MotionConfig(),now_s=1.2,clock_id='other',error_bound_m=0.)

    def test_invalid_and_stationary_candidates(self):
        c=self.candidate()
        with self.assertRaises(ValueError): replace(c,end=c.start)
        with self.assertRaises(ValueError): replace(c,start=[0.,0.,1.])
        with self.assertRaises(ValueError): stopping_volume(c,MotionConfig(),now_s=1.,error_bound_m=-.1)

    def test_planner_checks_alternatives_but_does_not_execute_assumed_free_path(self):
        r=planning_comparison(self.inspector(),origin=(-.5,0.,1.25),step_m=.5,start=(1,1),goal=(1,2),
            config=MotionConfig(),now_s=1.2,clock_id='clock',error_bound_m=.02)
        self.assertEqual(r['baseline_route'],[(1,1),(1,2)])
        self.assertEqual(len(r['alternatives']),4)
        self.assertEqual(r['alternatives'][0]['next_cell'],[1,2])
        self.assertEqual(r['selected_route'],[]); self.assertFalse(r['flight_authorized'])


if __name__=='__main__': unittest.main()
