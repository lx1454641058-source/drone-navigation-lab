"""来源：本项目原创。停止尾段/深度组合的身份、时效与完成复查。"""
from dataclasses import replace
import unittest

from drone_nav.async_stop import ForecastMonitor
from drone_nav.forecast_space import SimulationReference, query_stop_space, finish_stop_space
from drone_nav.stop_forecast import FrozenForecast, CLOCK
from drone_nav.pinhole import Intrinsics, Pose, PerspectiveFrame
from tools.forecast_space_experiment import build_inspector
import test_async_stop as async_fixture


class ForecastSpaceTests(unittest.TestCase):
    def fixture(self,z=5.):
        original_request,original_packet,_=async_fixture.AsyncStopTests().fixture()
        record=original_request.record()
        record['binding']['model_sha256']='a'*64
        request=FrozenForecast.create(record)
        packet_record=original_packet.record()
        packet_record['request_sha256']=request.sha256
        for point in packet_record['checkpoints']: point['model_sha256']='a'*64
        packet=FrozenForecast.create(packet_record)
        current=lambda i: {k:v for k,v in packet_record['checkpoints'][i//10].items() if k!='remaining'}
        monitor=ForecastMonitor(request)
        monitor.tick(current(100),elapsed_s=.2,packet=packet)
        k=Intrinsics(5,5,4.,4.,2.,2.)
        pose=Pose((-2.,0.,3.5),(0.,-1.,0.),(0.,0.,-1.),(1.,0.,0.))
        frame=PerspectiveFrame(k,pose,((0,0,0),)*25,(z,)*25,0)
        now=current(100)['time_s']
        inspector,reference,_=build_inspector(frame,capture_s=now-.1,frame_id='test-depth',model_sha='a'*64)
        return monitor,current,inspector,reference,now

    def query(self,fixture):
        monitor,current,inspector,reference,now=fixture
        return query_stop_space(monitor,current(100),inspector,reference,now_s=now,clock_id=CLOCK)

    def test_full_suffix_body_margin_and_opposing_errors_are_preserved(self):
        f=self.fixture(); q=self.query(f).record()
        self.assertEqual(q['volume']['lower'],[-.41,-.41,3.5-.37-.04])
        self.assertEqual(q['volume']['upper'],[.41,.41,3.5+.37+.04])
        self.assertEqual(q['horizon_end_s'],15.)
        self.assertTrue(q['expires_before_horizon'])
        self.assertEqual(q['geometry']['status'],'SAMPLED_BEYOND_VOLUME')
        self.assertFalse(q['flight_authorized'])

    def test_empty_fixture_detection_does_not_hide_depth_surface(self):
        q=self.query(self.fixture(z=2.)).record()
        self.assertEqual(q['geometry']['status'],'MEASURED_SURFACE')
        self.assertEqual(q['geometry']['semantic_surface_samples'],0)
        self.assertGreater(q['geometry']['inside_surface_pixels'],0)

    def test_unmatched_prediction_skips_geometric_query(self):
        f=self.fixture(); monitor,current,inspector,reference,now=f
        monitor.tick(dict(current(110),controller_sha256='b'*64),elapsed_s=.22)
        q=query_stop_space(monitor,current(110),inspector,reference,now_s=current(110)['time_s'],clock_id=CLOCK).record()
        self.assertIsNone(q['geometry'])
        self.assertIn('STOP_PREDICTION_NOT_MATCHED:EXECUTION_STATE_DIVERGED',q['reasons'])

    def test_matching_world_label_is_not_reference_identity(self):
        f=self.fixture(); m,c,i,r,t=f
        for bad,reason in ((replace(r,reference_fingerprint='b'*64),'CAMERA_REFERENCE_MISMATCH'),
                           (replace(r,model_sha256='b'*64),'MODEL_BINDING_MISMATCH'),
                           (replace(r,world_frame='foreign'),'WORLD_BINDING_MISMATCH'),
                           (replace(r,clock_id='foreign'),'CLOCK_BINDING_MISMATCH')):
            value=query_stop_space(m,c(100),i,bad,now_s=t,clock_id=CLOCK).record()
            self.assertIn(reason,value['reasons']); self.assertIsNone(value['geometry'])

    def test_altered_current_controller_state_is_rejected(self):
        m,c,i,r,t=self.fixture()
        bad=dict(c(100),controller_sha256='f'*64)
        q=query_stop_space(m,bad,i,r,now_s=t,clock_id=CLOCK).record()
        self.assertIn('CURRENT_CHECKPOINT_MISMATCH',q['reasons']); self.assertIsNone(q['geometry'])

    def test_completion_rechecks_expiry_and_clock(self):
        f=self.fixture(); m,c,i,r,t=f; q=self.query(f)
        result=finish_stop_space(q,m,c(100),i,now_s=t+.41,clock_id=CLOCK)
        self.assertIn('DEPTH_EXPIRED_DURING_QUERY',result['reasons']); self.assertFalse(result['result_current'])
        result=finish_stop_space(q,m,c(100),i,now_s=t,clock_id='other')
        self.assertIn('COMPLETION_CLOCK_MISMATCH',result['reasons'])

    def test_state_advance_invalidates_finished_query(self):
        f=self.fixture(); m,c,i,r,t=f; q=self.query(f)
        m.tick(c(110),elapsed_s=.22)
        result=finish_stop_space(q,m,c(110),i,now_s=c(110)['time_s'],clock_id=CLOCK)
        self.assertIn('STOP_CONTEXT_CHANGED_DURING_QUERY',result['reasons']); self.assertFalse(result['result_current'])

    def test_depth_replacement_invalidates_finished_query(self):
        f=self.fixture(); m,c,i,r,t=f; q=self.query(f)
        i.depths=(6.,)*25
        result=finish_stop_space(q,m,c(100),i,now_s=t,clock_id=CLOCK)
        self.assertIn('DEPTH_CONTEXT_CHANGED_DURING_QUERY',result['reasons'])

    def test_current_result_is_still_not_permission(self):
        f=self.fixture(); m,c,i,r,t=f
        result=finish_stop_space(self.query(f),m,c(100),i,now_s=t,clock_id=CLOCK)
        self.assertTrue(result['result_current'])
        for key in ('flight_authorized','selected_for_execution','free_volume_proven','navigation_map_update_allowed'):
            self.assertFalse(result[key])
        self.assertIn('STOP_MODEL_UNCERTAINTY_UNVALIDATED',result['reasons'])

    def test_foreign_clock_is_not_aged_or_hashed_as_local_depth(self):
        m,c,i,r,t=self.fixture()
        i.clock_id='foreign-relative-clock'
        i.available_at_s=1000.
        i.valid_until_s=1.
        # A rejected identity must not even attempt to hash this test sentinel.
        i.depths=object()
        q=query_stop_space(m,c(100),i,r,now_s=t,clock_id=CLOCK)
        result=finish_stop_space(q,m,c(100),i,now_s=t+.2,clock_id=CLOCK)
        self.assertIn('CLOCK_BINDING_MISMATCH',result['reasons'])
        self.assertNotIn('DEPTH_NOT_AVAILABLE',result['reasons'])
        self.assertNotIn('DEPTH_EXPIRED_DURING_QUERY',result['reasons'])
        self.assertIsNone(q.record()['depth_identity'])
        self.assertFalse(result['result_current'])
        # A caller supplying the same foreign clock as the depth must not bypass
        # the physical request's distinct clock identity.
        q=query_stop_space(m,c(100),i,r,now_s=t,clock_id=i.clock_id)
        result=finish_stop_space(q,m,c(100),i,now_s=t+.2,clock_id=i.clock_id)
        self.assertIn('CLOCK_BINDING_MISMATCH',result['reasons'])
        self.assertNotIn('DEPTH_NOT_AVAILABLE',result['reasons'])
        self.assertNotIn('DEPTH_EXPIRED_DURING_QUERY',result['reasons'])

    def test_bad_margin_tampered_content_and_fake_calibration_rejected(self):
        f=self.fixture(); m,c,i,r,t=f
        for margin in (-.01,float('nan'),3.):
            with self.assertRaises(ValueError): query_stop_space(m,c(100),i,r,now_s=t,clock_id=CLOCK,error_bound_m=margin)
        with self.assertRaises(ValueError): replace(r,kind='real-camera-calibrated')
        q=self.query(f)
        with self.assertRaises(ValueError): finish_stop_space(replace(q,content=q.content+' '),m,c(100),i,now_s=t,clock_id=CLOCK)
        with self.assertRaises(ValueError): finish_stop_space(q,m,c(100),i,now_s=t-.01,clock_id=CLOCK)


if __name__=='__main__': unittest.main()
