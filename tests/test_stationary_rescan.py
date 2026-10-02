"""来源：本项目原创。历史来源、原地观察动作和仍漏采的反例。"""
from dataclasses import replace
import unittest
from drone_nav.motion import MotionConfig
from drone_nav.pinhole import Intrinsics,Pose,PerspectiveFrame
from drone_nav.sampling_motion import SamplingView,SamplingAssumption
from drone_nav.stationary_rescan import SamplingHistory,next_phase,phase_pose
from tools.stationary_rescan_experiment import specs,simulate_case


def view(tick=0,t=0.,x=2.5):
    k=Intrinsics(40,30,25,25,19.5,14.5)
    f=PerspectiveFrame(k,Pose.look_at((x,5.5,3.5),(x+1,5.5,3.5)),((100,100,100),)*1200,(9.5,)*1200,tick)
    return SamplingView(f,'frame-'+str(tick),'camera','simulation-seconds',t,t+.1)


class StationaryRescanTests(unittest.TestCase):
    def guard(self,h,now):return h.guard((4,5),(5,5),now_s=now,config=MotionConfig(),assumption=SamplingAssumption(.2,.2))

    def test_new_batch_keeps_independent_old_support(self):
        h=SamplingHistory();h.add(view(),now_s=.1);h.add(view(1,1.,4.5),now_s=1.1)
        self.assertEqual(h.grid.free_seen[(4,5,3)],1)
        r=self.guard(h,1.1);self.assertTrue(r['allowed'])
        old=r['sampling'][0]['candidates'][0];self.assertEqual(old['captured_at_s'],0);self.assertEqual(old['valid_until_s'],12)
        self.assertFalse(old['reasons'])

    def test_new_occupied_overrides_old_free(self):
        h=SamplingHistory();h.add(view(),now_s=.1);v=view(1,1.,4.5);z=list(v.frame.depth_z_m);z[14*40+19]=.7
        h.add(replace(v,frame=replace(v.frame,depth_z_m=z)),now_s=1.1)
        r=self.guard(h,1.1);self.assertFalse(r['allowed']);self.assertFalse(r['baseline']['allowed'])

    def test_refresh_does_not_extend_old_coverage_age(self):
        h=SamplingHistory();h.add(view(),now_s=.1);h.add(view(1,9.9,4.5),now_s=10.)
        r=self.guard(h,10);self.assertTrue(r['baseline']['allowed']);self.assertFalse(r['allowed'])
        self.assertIn('SAMPLING_EXPIRES_BEFORE_STOP',r['sampling'][0]['candidates'][0]['reasons'])

    def test_bad_frame_atomic_and_duplicate_rejected(self):
        h=SamplingHistory();v=view();h.add(v,now_s=.1);before=dict(h.grid.free_seen)
        for bad in (v,replace(view(1,1.,4.5),clock_id='wrong'),
                    replace(view(1,1.,4.5),frame=replace(view(1,1.,4.5).frame,depth_z_m=(1000.,)*1200))):
            with self.assertRaises(ValueError):h.add(bad,now_s=1.1)
            self.assertEqual(h.grid.free_seen,before);self.assertEqual(h.grid.last_tick,0)
        self.assertTrue(self.guard(h,.2)['allowed'])

    def test_consumer_cannot_rewind_and_stored_frame_detached(self):
        h=SamplingHistory();v=view();z=list(v.frame.depth_z_m);h.add(replace(v,frame=replace(v.frame,depth_z_m=z)),now_s=.1)
        z[0]=float('nan');self.assertTrue(self.guard(h,.2)['allowed'])
        with self.assertRaises(ValueError):self.guard(h,.1)

    def test_phase_order_and_stationary_position(self):
        used=[]
        for _ in range(4):used.append(next_phase(used,'phase_diverse'))
        self.assertEqual(used,[(0,0),(.5,.5),(.5,0),(0,.5)])
        with self.assertRaises(ValueError):next_phase(used,'phase_diverse')
        self.assertEqual(next_phase(used,'repeat'),(0,0))
        k=Intrinsics(40,30,25,25,19.5,14.5)
        for phase in used:self.assertEqual(phase_pose((4,5),(5,5),k,phase).position,(4.5,5.5,3.5))
        with self.assertRaises(ValueError):phase_pose((4,5),(5,6),k,(0,0))

    def case(self,key,mode='phase_diverse'):
        s=next(s for s in specs() if s['key']==key);return simulate_case(s,mode)

    def test_phase_change_finds_original_hidden_obstacle(self):
        raw,r=self.case('hidden-20mm');self.assertEqual(r['state'],'OBSERVED_OBSTACLE_HOLD')
        self.assertEqual(len(raw['scans']),2);self.assertGreater(raw['scans'][1]['thin_pixels'],0)
        self.assertIsNone(r['motion']);self.assertTrue(r['initial_audit']['collision'])
        _,fixed=self.case('hidden-20mm','repeat');self.assertTrue(fixed['final_audit']['collision'])

    def test_empty_history_supports_action_after_four_scans(self):
        raw,r=self.case('empty');self.assertEqual(len(raw['scans']),4)
        self.assertEqual(r['state'],'READY_FOR_SIMULATED_MOVE');self.assertAlmostEqual(r['elapsed_scan_s'],.6)
        self.assertFalse(r['final_audit']['collision'])

    def test_unknown_size_and_missing_history_never_fabricated(self):
        for key in ('empty-unknown','no-history','expired-history','slow-scan'):
            _,r=self.case(key);self.assertIsNone(r['motion']);self.assertEqual(r['state'],'SCAN_BUDGET_HOLD')

    def test_failed_scan_stops_even_if_previous_guard_allowed(self):
        for key,state in [('depth-loss','SCAN_SENSOR_HOLD'),('clock-fault','SCAN_PROTOCOL_HOLD')]:
            raw,r=self.case(key);self.assertEqual(len(raw['scans']),2);self.assertEqual(r['state'],state)
            self.assertIsNone(r['motion'])

    def test_remaining_thinner_counterexample_not_hidden(self):
        raw,r=self.case('hidden-2mm-offset');self.assertEqual(sum(s['thin_pixels'] for s in raw['scans']),0)
        self.assertTrue(r['final_audit']['collision']);self.assertFalse(r['flight_authorized'])

    def test_saved_inputs_replay_policy_and_history_exactly(self):
        raw,r=self.case('hidden-20mm');saved,again=simulate_case(raw['spec'],raw['mode'],raw)
        self.assertEqual(raw,saved);self.assertEqual(r,again)


if __name__=='__main__':unittest.main()
