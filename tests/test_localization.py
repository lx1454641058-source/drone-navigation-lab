"""来源：本项目原创。定位误差、任意方向线段评价及到达误报回归测试。"""

from dataclasses import asdict,replace
import gzip
import hashlib
import json
import math
from pathlib import Path
from random import Random
import tempfile
import unittest

from drone_nav.exploration import scan_poses
from drone_nav.exploration_experiment import segment_box_distance
from drone_nav.localization import (LocalizationBudget,MislocalizedCamera,PositionError,
                                    audit_localization,segment_box_clearance)
from drone_nav.localization_experiment import localization_cases,summarize
from drone_nav.motion import MotionConfig,movement_guard,profile
from drone_nav.occupancy import VoxelMap
from drone_nav.pinhole import Intrinsics
from drone_nav.raycast import Box,render
from drone_nav.timed_navigation import navigate
from drone_nav.verify_localization import verify


class ErrorBudgetTests(unittest.TestCase):
    def test_arrival_uses_radial_not_single_axis_error(self):
        self.assertTrue(LocalizationBudget(.3).arrival_confirmable())
        self.assertFalse(LocalizationBudget(.45).arrival_confirmable())

    def test_margin_grows_when_error_exceeds_existing_buffer(self):
        self.assertEqual(LocalizationBudget(.25).planning_margin(.25),1)
        self.assertEqual(LocalizationBudget(.8).planning_margin(.25),2)

    def test_invalid_budget_and_error_model(self):
        for value in (-1,3,math.nan,True):
            with self.assertRaises(ValueError): LocalizationBudget(value)
            with self.assertRaises(ValueError): PositionError(value)
        with self.assertRaises(ValueError): LocalizationBudget(0,0)
        with self.assertRaises(ValueError): PositionError(.1,'unknown')
        with self.assertRaises(ValueError): PositionError(.1,'drift',1)

    def test_drift_is_within_declared_axis_bound(self):
        model = PositionError(.45,'drift')
        for tick in range(100):
            self.assertLessEqual(max(abs(x) for x in model.at(tick)),.45)
        self.assertNotEqual(model.at(0),model.at(1))

    def test_relative_error_guard_rejects_side_unknown(self):
        grid = VoxelMap(10,10)
        grid.free_seen = {(x,y,3):0 for x in range(10) for y in range(10)}
        del grid.free_seen[(4,3,3)]
        c = MotionConfig()
        args = (grid,(3,4),(4,4),0,.9,{0:0},profile(1,c),c)
        self.assertTrue(movement_guard(*args)['allowed'])
        self.assertFalse(movement_guard(*args,error_bound_m=.25)['allowed'])

    def test_frame_contains_true_pixels_but_only_estimated_pose(self):
        case = localization_cases()[1]
        pose = scan_poses((3.5,8.5,3.5),(16,8))[0]
        k = Intrinsics(8,6,5,5,3.5,2.5)
        reported = MislocalizedCamera(case['world'],case['error']).capture(pose,k,0)
        offset = case['error'].at(0)
        true_pose = replace(pose,position=tuple(a+b for a,b in zip(pose.position,offset)))
        actual,_ = render(case['world'],true_pose,k,tick=0,seed=1201)
        self.assertEqual(reported.pose,pose)
        self.assertNotEqual(reported.pose,true_pose)
        self.assertEqual(reported.rgb,actual.rgb)
        self.assertEqual(reported.depth_z_m,actual.depth_z_m)


class ContinuousGeometryTests(unittest.TestCase):
    def setUp(self):
        self.box = Box('test',3,(0,0,0),(1,1,1))

    def test_segment_through_box(self):
        self.assertEqual(segment_box_clearance((-1,2,.5),(2,-1,.5),self.box),0)

    def test_closest_point_inside_segment_near_corner(self):
        self.assertAlmostEqual(segment_box_clearance((0,2.5,.5),(2.5,0,.5),self.box),math.sqrt(.125))

    def test_parallel_and_point(self):
        self.assertEqual(segment_box_clearance((-1,2,.5),(2,2,.5),self.box),1)
        self.assertAlmostEqual(segment_box_clearance((2,2,2),(2,2,2),self.box),math.sqrt(3))

    def test_axis_segments_match_prior_independent_evaluator(self):
        for a,b in (((-1,2,.5),(3,2,.5)),((2,-2,2),(2,3,2)),((2,2,-1),(2,2,4))):
            self.assertAlmostEqual(segment_box_clearance(a,b,self.box),segment_box_distance(a,b,self.box))

    def test_exact_minimum_is_bounded_by_dense_sample_and_lipschitz_gap(self):
        rng = Random(62026)
        for _ in range(30):
            a,b = [tuple(rng.uniform(-3,3) for _ in range(3)) for _ in range(2)]
            exact = segment_box_clearance(a,b,self.box)
            sampled = min(math.sqrt(sum((v-min(1,max(0,v)))**2 for v in
                                       [x+(y-x)*i/500 for x,y in zip(a,b)])) for i in range(501))
            segment_length = math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
            self.assertLessEqual(exact,sampled+1e-10)
            self.assertGreaterEqual(exact,sampled-segment_length/1000-1e-10)

    def test_invalid_endpoints(self):
        with self.assertRaises(ValueError): segment_box_clearance((math.nan,0,0),(1,1,1),self.box)


class LocalizationNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = localization_cases()[:4]
        cls.results = []
        for case in cls.cases:
            camera = MislocalizedCamera(case['world'],case['error'])
            r = navigate(camera,20,16,case['start'],case['goal'],localization=case['budget'],previews=False)
            r['audit'] = audit_localization(case['world'],r,case['error'],case['budget'])
            cls.results.append(r)

    def test_ideal_goal_matches_truth(self):
        r = self.results[0]
        self.assertEqual(r['terminal_state'],'ARRIVED_WAYPOINT')
        self.assertEqual(r['audit']['final_goal_error_m'],0)

    def test_unaccounted_bias_causes_false_arrival(self):
        r = self.results[1]
        self.assertEqual(r['terminal_state'],'ARRIVED_WAYPOINT')
        self.assertTrue(r['audit']['false_arrival'])
        self.assertAlmostEqual(r['audit']['final_goal_error_m'],math.sqrt(2)*.45)
        self.assertGreater(r['audit']['declared_bound_violations'],0)

    def test_declared_bias_prevents_false_arrival(self):
        r = self.results[2]
        self.assertEqual(r['terminal_state'],'UNCERTAIN_ARRIVAL')
        self.assertFalse(r['audit']['false_arrival'])
        self.assertEqual(r['audit']['declared_bound_violations'],0)

    def test_drift_keeps_continuous_true_path(self):
        r = self.results[3]
        moves = r['audit']['true_moves']
        self.assertTrue(all(a['to']==b['from'] for a,b in zip(moves,moves[1:])))
        self.assertEqual(moves[-1]['to'],r['audit']['true_final'])
        self.assertTrue(r['audit']['inside_goal_tolerance'])
        self.assertEqual(r['audit']['declared_bound_violations'],0)

    def test_error_truth_is_not_in_navigation_output(self):
        # 真值只由运行后的独立评价补入；导航 trace 没有实际偏移或真实位置。
        for frame in self.results[3]['trace']:
            self.assertNotIn('true_position',frame)
            self.assertNotIn('actual_offset',frame)


class LocalizationReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = localization_cases()[3]
        c = cls.case
        camera = MislocalizedCamera(c['world'],c['error'],record=True)
        cls.result = navigate(camera,20,16,c['start'],c['goal'],localization=c['budget'],max_ticks=1,previews=False)
        cls.result.update(key=c['key'],condition=c['condition'],world=asdict(c['world']),
                          error_model=asdict(c['error']),audit=audit_localization(c['world'],cls.result,c['error'],c['budget']))
        cls.blob = gzip.compress(json.dumps(camera.frames).encode('utf-8'),mtime=0)

    def fixture(self,folder):
        root = Path(folder)
        name = self.case['key']+'-observations.json.gz'
        (root/name).write_bytes(self.blob)
        report = json.loads(json.dumps({'results':[self.result],'summary':summarize([self.result])}))
        (root/'localization_report.json').write_text(json.dumps(report),encoding='utf-8')
        (root/'observations_manifest.json').write_text(json.dumps([
            {'path':name,'frames':8,'sha256':hashlib.sha256(self.blob).hexdigest()}]),encoding='utf-8')
        return root,report

    def test_raw_frames_and_true_trajectory_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root,_ = self.fixture(folder)
            self.assertTrue(verify(root)['verified'])

    def test_modified_true_trajectory_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,report = self.fixture(folder)
            report['results'][0]['audit']['true_final'][0] += 1
            (root/'localization_report.json').write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'geometry'): verify(root)

    def test_modified_summary_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,report = self.fixture(folder)
            report['summary'][0]['runs'] = 99
            (root/'localization_report.json').write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'summary'): verify(root)


if __name__=='__main__': unittest.main()
