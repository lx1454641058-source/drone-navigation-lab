"""来源：本项目原创。解析几何、语义反例、异常观测与两阶段门控验证。"""

from dataclasses import asdict,replace
from math import nan
import unittest
from unittest.mock import patch

from drone_nav.perspective_landing import LandingConfig,inspect_landing
from drone_nav.pinhole import Intrinsics,Pose
from drone_nav.surface_geometry import fit_plane
from drone_nav.verify_delivery import RecordedDeliveryCamera,normalized
from drone_nav.vision_experiment import train_model
from drone_nav.visual_delivery import run_visual_delivery
from drone_nav.visual_delivery_experiment import VisualDeliveryCamera,delivery_cases


class LandingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train_model()
        cls.pose = Pose.look_at((16.5,8.5,3.5),(16.5,8.5,0))
        cls.k = Intrinsics(96,72,45,45,47.5,35.5)
        cls.cases = {c['key']:c for c in delivery_cases()}

    def frame(self,key='paved'):
        c = self.cases[key]
        return VisualDeliveryCamera(c['world'],light=c['light'],dropout=c['dropout']).capture_landing(
            self.pose,c.get('intrinsics',self.k),1)

    def inspect(self,frame):
        return inspect_landing(frame,self.model,(16.5,8.5,0),expected_tick=1,expected_position=self.pose.position)[0]

    def test_flat_and_analytic_slopes(self):
        for key,slope,accepted in [('paved',0,True),('gentle',3,True),('steep',8,False)]:
            with self.subTest(key=key):
                e = self.inspect(self.frame(key))
                self.assertEqual(e['accepted'],accepted)
                self.assertAlmostEqual(e['slope_deg'],slope,places=8)
                self.assertLess(e['roughness_m'],1e-10)

    def test_water_and_vegetation_geometry_cannot_detect(self):
        for key in ('water','vegetation'):
            e = self.inspect(self.frame(key))
            self.assertTrue(e['geometry_accepted'])
            self.assertFalse(e['semantic_accepted'])
            self.assertFalse(e['accepted'])

    def test_rgb_only_counterfactual_changes_decision(self):
        paved,water = self.frame(),self.frame('water')
        changed = replace(paved,rgb=water.rgb)
        a,b = self.inspect(paved),self.inspect(changed)
        self.assertEqual(paved.depth_z_m,changed.depth_z_m)
        self.assertEqual(a['slope_deg'],b['slope_deg'])
        self.assertTrue(a['accepted'])
        self.assertFalse(b['accepted'])

    def test_same_color_bump_requires_geometry(self):
        e = self.inspect(self.frame('bump'))
        self.assertTrue(e['semantic_accepted'])
        self.assertIn('UNEVEN_SURFACE',e['reason_codes'])
        self.assertGreater(e['max_residual_m'],.15)

    def test_person_occlusion_rejected(self):
        e = self.inspect(self.frame('person'))
        self.assertFalse(e['accepted'])
        self.assertGreater(e['class_counts'].get('person',0),0)
        self.assertIn('UNEVEN_SURFACE',e['reason_codes'])

    def test_dark_and_unknown_rejected(self):
        for key in ('dark','unknown'):
            e = self.inspect(self.frame(key))
            self.assertTrue(e['geometry_accepted'])
            self.assertFalse(e['accepted'])
            self.assertGreater(e['class_counts'].get('unknown',0),0)

    def test_missing_depth_inside_footprint_rejected(self):
        frame = self.frame()
        depths = list(frame.depth_z_m)
        depths[35*96+47] = None
        e = self.inspect(replace(frame,depth_z_m=tuple(depths)))
        self.assertIn('MISSING_DEPTH',e['reason_codes'])
        self.assertEqual(e['missing_depth'],1)

    def test_missing_depth_outside_footprint_does_not_reject(self):
        frame = self.frame()
        e = self.inspect(replace(frame,depth_z_m=(None,)+frame.depth_z_m[1:]))
        self.assertTrue(e['accepted'])

    def test_narrow_view_rejected(self):
        self.assertIn('INCOMPLETE_COVERAGE',self.inspect(self.frame('narrow'))['reason_codes'])

    def test_stale_or_wrong_position_rejected(self):
        with self.assertRaises(ValueError): self.inspect(replace(self.frame(),tick=0))
        frame = self.frame()
        with self.assertRaises(ValueError):
            self.inspect(replace(frame,pose=replace(frame.pose,position=(16.6,8.5,3.5))))

    def test_plane_solver_rejects_degenerate_or_nonfinite_samples(self):
        for points in ([],[(0,0,0)]*5,[(0,0,0),(1,1,1),(2,2,2)],[(0,0,0),(1,0,0),(0,1,nan)]):
            with self.assertRaises(ValueError): fit_plane(points)

    def test_elevated_flat_surface_rejected(self):
        frame = self.frame()
        e = self.inspect(replace(frame,depth_z_m=tuple(d-.3 for d in frame.depth_z_m)))
        self.assertIn('UNEXPECTED_ELEVATION',e['reason_codes'])

    def test_invalid_configuration(self):
        for settings in ({'radius_m':0},{'max_slope_deg':45},{'min_samples':True},{'max_residual_m':nan}):
            with self.assertRaises(ValueError): LandingConfig(**settings)

    def test_navigation_failure_never_captures_landing(self):
        class NoLanding:
            def capture_landing(self,*args): raise AssertionError('must not capture')
        with patch('drone_nav.visual_delivery.navigate',return_value={'terminal_state':'NO_FEASIBLE_FRONTIER'}):
            r = run_visual_delivery(NoLanding(),self.model,(3,8),(16,8),previews=False)
        self.assertEqual(r['terminal_state'],'NAVIGATION_HOLD')
        self.assertIsNone(r['landing'])

    def test_camera_calibration_mismatch_holds(self):
        frame = self.frame('narrow')
        class WrongCalibration:
            def capture_landing(self,*args): return frame
        nav = dict(terminal_state='ARRIVED_WAYPOINT',trace=[{'position':(16,8)}],altitude_m=3.5,observations=1)
        with patch('drone_nav.visual_delivery.navigate',return_value=nav):
            r = run_visual_delivery(WrongCalibration(),self.model,(3,8),(16,8),previews=False)
        self.assertEqual(r['terminal_state'],'LANDING_SENSOR_HOLD')

    def test_invalid_reference_rejected_before_navigation(self):
        for value in (nan,3.5,True):
            with self.assertRaises(ValueError):
                run_visual_delivery(None,self.model,(3,8),(16,8),reference_z_m=value)

    def test_real_navigation_and_saved_pixel_replay(self):
        camera = VisualDeliveryCamera(self.cases['water']['world'],record=True)
        result = run_visual_delivery(camera,self.model,(3,8),(4,8),previews=False)
        self.assertEqual(result['navigation']['terminal_state'],'ARRIVED_WAYPOINT')
        self.assertEqual(result['terminal_state'],'LANDING_REJECTED')
        recorded = RecordedDeliveryCamera(dict(navigation=camera.frames,landing=camera.landing_frames))
        replay = run_visual_delivery(recorded,self.model,(3,8),(4,8),previews=False)
        self.assertEqual(normalized(result),normalized(replay))
        self.assertEqual(recorded.index,len(camera.frames))
        self.assertEqual(recorded.landing_camera.index,1)


if __name__=='__main__': unittest.main()
