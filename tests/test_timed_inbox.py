"""来源：本项目原创。接收/消费边界、队列故障、输入所有权与坐标隔离。"""
from copy import deepcopy
import unittest
from drone_nav.timed_inbox import TimedObservationInbox


def result(frame='f1',capture=1.,samples=True):
    observations=[] if not samples else [dict(samples=[dict(pixel=[2,3],depth_z_m=2.,world_m=[1.,2.,3.])],
                                              association='unverified_box_surface',object_extent_known=False)]
    return dict(flight_authorized=False,navigation_frame_alignment_available=False,
        reason='VISUAL_TARGET_HOLD' if samples else 'NO_DETECTION_REQUIRES_GEOMETRY',
        projection=dict(frame_id=frame,camera_id='camera',clock_id='clock',consumer_clock_id='clock',
            detector_source='model',captured_at_s=capture,completed_at_s=capture+.1,
            consumer_at_s=capture+.2,valid_until_s=capture+.5,age_s=.2,
            world_frame='source-world',status='SAMPLES_AVAILABLE' if samples else 'NO_SPATIAL_SAMPLES',
            reasons=[],observations=observations,free_space_evidence=[],flight_authorized=False))


class InboxTests(unittest.TestCase):
    def inbox(self,capacity=2):
        return TimedObservationInbox(camera_id='camera',clock_id='clock',world_frame='source-world',capacity=capacity)

    def submit(self,inbox,r,now): return inbox.submit(r,now_s=now,clock_id='clock')
    def consume(self,inbox,now,world='source-world'):
        return inbox.consume(now_s=now,clock_id='clock',target_world=world)

    def test_deadline_inclusive_but_next_instant_expires(self):
        inbox=self.inbox(); self.submit(inbox,result(),1.2)
        self.assertTrue(self.consume(inbox,1.5)['delivered'])
        inbox=self.inbox(); self.submit(inbox,result(),1.2)
        self.assertEqual(self.consume(inbox,1.50001)['reason'],'EXPIRED_AT_CONSUMPTION')

    def test_delayed_arrival_and_future_result_rejected(self):
        inbox=self.inbox()
        self.assertEqual(self.submit(inbox,result(),1.19)['reason'],'RESULT_NOT_READY')
        self.assertEqual(self.submit(inbox,result(),1.51)['reason'],'EXPIRED_AT_RECEIPT')
        self.assertEqual(inbox.status()['pending'],0)

    def test_duplicate_and_out_of_order_do_not_refresh_pending(self):
        inbox=self.inbox(); self.submit(inbox,result(),1.2)
        self.assertEqual(self.submit(inbox,result(),1.3)['reason'],'REPEATED_FRAME')
        self.assertEqual(self.submit(inbox,result('older',.9),1.3)['reason'],'OUT_OF_ORDER_CAPTURE')
        self.assertEqual(self.consume(inbox,1.6)['reason'],'EXPIRED_AT_CONSUMPTION')

    def test_overflow_latches_and_cannot_be_recovered_by_empty_result(self):
        inbox=self.inbox(capacity=1); self.submit(inbox,result(),1.2)
        self.assertEqual(self.submit(inbox,result('f2',1.1),1.3)['reason'],'QUEUE_OVERFLOW')
        self.assertEqual(self.consume(inbox,1.3)['reason'],'STREAM_FAULT_LATCHED')
        self.assertEqual(self.submit(inbox,result('empty',1.2,False),1.4)['reason'],'STREAM_FAULT_LATCHED')
        self.assertEqual(inbox.status()['pending'],1)

    def test_empty_frame_cannot_displace_queued_positive_or_clear_evidence(self):
        inbox=self.inbox(); self.submit(inbox,result(),1.2)
        self.submit(inbox,result('empty',1.1,False),1.3)
        first=self.consume(inbox,1.3); empty=self.consume(inbox,1.3)
        self.assertEqual(first['reason'],'VISUAL_TARGET_HOLD')
        self.assertEqual(empty['reason'],'NO_DETECTION_REQUIRES_GEOMETRY')
        self.assertFalse(empty['clears_previous_evidence']); self.assertFalse(empty['navigation_map_update_allowed'])
        self.assertEqual(empty['free_space_evidence'],[])

    def test_wrong_target_keeps_queue_and_never_relabels_points(self):
        inbox=self.inbox(); self.submit(inbox,result(),1.2)
        wrong=self.consume(inbox,1.3,'navigation-world')
        self.assertEqual(wrong['reason'],'TARGET_WORLD_MISMATCH'); self.assertEqual(wrong['pending'],1)
        correct=self.consume(inbox,1.3)
        self.assertEqual(correct['world_frame'],'source-world')

    def test_wrong_camera_world_and_source_clock_rejected(self):
        for key,value,reason in [('camera_id','other','CAMERA_MISMATCH'),('world_frame','other','SOURCE_WORLD_MISMATCH'),
                                 ('clock_id','other','SOURCE_CLOCK_MISMATCH')]:
            r=result(); r['projection'][key]=value
            self.assertEqual(self.submit(self.inbox(),r,1.2)['reason'],reason)

    def test_mutable_caller_payload_is_detached(self):
        inbox=self.inbox(); r=result(); self.submit(inbox,r,1.2)
        r['projection']['observations'][0]['samples'][0]['world_m'][0]=999
        self.assertEqual(self.consume(inbox,1.3)['observation']['projection']['observations'][0]['samples'][0]['world_m'][0],1.)

    def test_malformed_input_and_clock_errors_are_atomic(self):
        inbox=self.inbox(); before=inbox.status()
        for mutate in (lambda p:p.update(valid_until_s=99),lambda p:p.update(age_s=9),
                       lambda p:p.update(flight_authorized=True),lambda p:p.update(free_space_evidence=['free']),
                       lambda p:p['observations'][0]['samples'][0].update(world_m=[float('nan'),0,0])):
            r=result(); mutate(r['projection'])
            with self.assertRaises(ValueError): self.submit(inbox,r,1.2)
            self.assertEqual(inbox.status(),before)
        with self.assertRaises(ValueError): inbox.consume(now_s=1,clock_id='wrong',target_world='source-world')
        self.assertEqual(inbox.status(),before)
        self.submit(inbox,result(),1.2); before=inbox.status()
        with self.assertRaises(ValueError): self.consume(inbox,1.1)
        self.assertEqual(inbox.status(),before)

    def test_capacity_rejects_bool_zero_and_unbounded_settings(self):
        for capacity in (True,0,65,1.5):
            with self.assertRaises(ValueError): self.inbox(capacity)


if __name__=='__main__': unittest.main()
