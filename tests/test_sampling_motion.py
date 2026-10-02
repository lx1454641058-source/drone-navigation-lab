"""来源：本项目原创。组合运动检查的拒绝单调性与边界。"""
from dataclasses import replace
import unittest
from drone_nav.motion import MotionConfig,profile
from drone_nav.occupancy import VoxelMap
from drone_nav.pinhole import Intrinsics,Pose,PerspectiveFrame
from drone_nav.sampling_motion import SamplingAssumption,SamplingView,sampled_movement_guard


def fixture(width=40):
    k=Intrinsics(width,width*3//4,25*width/40,25*width/40,width/2-.5,width*3/8-.5)
    f=PerspectiveFrame(k,Pose.look_at((2.5,5.5,3.5),(3.5,5.5,3.5)),((100,100,100),)*(k.width*k.height),
                       (9.5,)*(k.width*k.height),0)
    g=VoxelMap(20,16,8,100);g.free_seen={(x,5,3):0 for x in range(3,9)};g.last_tick=0
    return g,SamplingView(f,'frame-0','camera','sim',0,.1)


class SamplingMotionTests(unittest.TestCase):
    def guard(self,g,v,assumption=SamplingAssumption(.2,.2),config=None,now=1,**kwargs):
        config=config or MotionConfig()
        return sampled_movement_guard(g,(4,5),(5,5),0,now,{0:0},profile(1,config),config,
                                      views=v if isinstance(v,list) else [v],assumption=assumption,clock_id='sim',**kwargs)

    def test_unknown_assumption_holds_and_keeps_base_result(self):
        g,v=fixture();r=self.guard(g,v,assumption=None)
        self.assertTrue(r['baseline']['allowed']);self.assertFalse(r['allowed'])
        self.assertIn('MINIMUM_FEATURE_ASSUMPTION_UNKNOWN',r['sampling'][0]['views'][0]['reasons'])

    def test_pitch_uses_all_stopping_cells_not_endpoint_only(self):
        g,v=fixture();r=self.guard(g,v)
        self.assertTrue(r['allowed']);self.assertEqual(r['required_cells'],[[4,5,3],[5,5,3],[6,5,3]])
        self.assertAlmostEqual(r['sampling'][-1]['views'][0]['pitch_m'][0],.18)
        r=self.guard(g,v,config=MotionConfig(distance_margin_m=1.2))
        self.assertTrue(r['baseline']['allowed']);self.assertFalse(r['allowed'])
        self.assertIn([7,5,3],r['required_cells'])
        self.assertIn('SAMPLING_TOO_COARSE',r['sampling'][-1]['views'][0]['reasons'])

    def test_resolution_changes_pitch_with_same_view(self):
        for width,allowed in ((40,False),(160,True)):
            g,v=fixture(width);self.assertEqual(self.guard(g,v,SamplingAssumption(.1,.1))['allowed'],allowed)

    def test_original_occupied_unknown_and_stale_never_reopened(self):
        for mode in ('occupied','unknown','stale'):
            g,v=fixture()
            if mode=='occupied':g.occupied.add((5,5,3))
            if mode=='unknown':g.free_seen.pop((5,5,3))
            r=self.guard(g,v,now=10 if mode=='stale' else 1)
            self.assertFalse(r['baseline']['allowed']);self.assertFalse(r['allowed'])
            self.assertEqual(r['reason'],r['baseline']['reason'])

    def test_camera_inside_required_voxel_cannot_cover_it(self):
        g,v=fixture();v=replace(v,frame=replace(v.frame,pose=Pose.look_at((4.5,5.5,3.5),(5.5,5.5,3.5))))
        r=self.guard(g,v);self.assertFalse(r['allowed'])
        self.assertIn('VOXEL_NOT_FULLY_IN_FRONT',r['sampling'][0]['views'][0]['reasons'])

    def test_missing_one_pixel_in_window_holds(self):
        g,v=fixture();depth=list(v.frame.depth_z_m);depth[14*40+19]=None
        r=self.guard(g,replace(v,frame=replace(v.frame,depth_z_m=tuple(depth))))
        self.assertTrue(r['baseline']['allowed']);self.assertFalse(r['allowed'])
        self.assertIn('MISSING_DEPTH',r['sampling'][0]['views'][0]['reasons'])

    def test_depth_boundary_and_range(self):
        g,v=fixture()
        for depth,reason in ((4.52,'FOREGROUND_OR_DEPTH_MARGIN'),(100.,'OUT_OF_RANGE')):
            r=self.guard(g,replace(v,frame=replace(v.frame,depth_z_m=(depth,)*1200)))
            self.assertFalse(r['allowed']);self.assertIn(reason,r['sampling'][-1]['views'][0]['reasons'])

    def test_clock_availability_and_map_time_identity(self):
        g,v=fixture()
        for bad,reason in ((replace(v,clock_id='other'),'CLOCK_MISMATCH'),
                           (replace(v,available_at_s=2),'FRAME_NOT_AVAILABLE'),
                           (replace(v,captured_at_s=.05),'MAP_CAPTURE_TIME_MISMATCH'),
                           (replace(v,frame=replace(v.frame,tick=1)),'MAP_EVIDENCE_BATCH_MISMATCH')):
            r=self.guard(g,bad);self.assertFalse(r['allowed'])
            self.assertIn(reason,r['sampling'][0]['views'][0]['reasons'])

    def test_duplicate_frames_and_malformed_input_fail(self):
        g,v=fixture()
        with self.assertRaises(ValueError):self.guard(g,[v,v])
        with self.assertRaises(ValueError):SamplingAssumption(True,.1)
        with self.assertRaises(ValueError):SamplingAssumption(.1,.1,'arbitrary_shapes')
        with self.assertRaises(ValueError):self.guard(g,replace(v,frame=replace(v.frame,depth_z_m=(float('nan'),)*1200)))

    def test_views_cannot_inflate_support_or_mutate_map(self):
        g,v=fixture();old=dict(g.free_seen)
        self.assertFalse(self.guard(g,[])['allowed'])
        r=self.guard(g,[replace(v,clock_id='wrong',frame_id='bad'),v])
        self.assertTrue(r['allowed']);self.assertEqual(g.free_seen,old)
        self.assertFalse(r['flight_authorized'])

    def test_hidden_feature_wrong_assumption_remains_a_counterexample(self):
        from tools.sampling_motion_experiment import specs,prepare,replay
        cases={s['key']:s for s in specs()}
        empty=prepare(cases['adequate-20cm']);hidden=prepare(cases['thin-wrong-assumption'])
        # 完全相同的中心 RGB-D 可对应有/无细障碍两个世界，不能凭这些像素区分。
        self.assertEqual(empty['frame'],hidden['frame']);self.assertEqual(hidden['thin_pixels'],0)
        r=replay(hidden);self.assertTrue(r['guard']['allowed']);self.assertTrue(r['combined_audit']['collision'])
        for key in ('thin-correct-assumption','thin-unknown-assumption'):
            r=replay(prepare(cases[key]))
            self.assertTrue(r['baseline_audit']['collision']);self.assertFalse(r['guard']['allowed'])
            self.assertIsNone(r['combined_motion'])


if __name__=='__main__':unittest.main()
