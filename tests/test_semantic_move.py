"""来源：本项目原创。模型输出只能增加拒绝；时间与相机身份不能跳过。"""
from dataclasses import replace
from pathlib import Path
import unittest

from drone_nav.pinhole import Intrinsics, Pose, PerspectiveFrame
from drone_nav.semantic_sensor import packet_for, context_for, visual_move_decision
from drone_nav.semantic_move import SemanticMoveMixin
from tools.tinyformer_probe import MODEL_SHA


class SemanticDecisionTests(unittest.TestCase):
    def setUp(self):
        k=Intrinsics(8,6,5,5,3.5,2.5)
        self.frame=PerspectiveFrame(k,Pose.look_at((3.5,8.5,3.5),(4.5,8.5,.8)),
            ((80,80,80),)*48,(3.,)*48,0)
        self.record=dict(model_sha256=MODEL_SHA,result=dict(boxes=[]))

    def decision(self, record=None, context=True, completed=.3, now=.3):
        p=packet_for(record or self.record,self.frame,frame_id='a',captured_at_s=0,completed_at_s=completed)
        c=context_for(self.frame,'a',0) if context is True else context
        return visual_move_decision(p,c,now_s=now)

    def test_empty_detection_still_requires_original_geometry(self):
        r=self.decision()
        self.assertTrue(r['permit_geometry_check'])
        self.assertFalse(r['flight_authorized'])
        self.assertEqual(r['projection']['free_space_evidence'],[])

    def test_recognized_target_is_veto_even_with_projectable_surface(self):
        record=dict(self.record,result=dict(boxes=[dict(group='person',score=.8,x1=2,y1=1,x2=6,y2=5)]))
        r=self.decision(record)
        self.assertEqual(r['reason'],'VISUAL_TARGET_HOLD')
        self.assertFalse(r['permit_geometry_check'])
        self.assertGreater(len(r['projection']['observations'][0]['samples']),0)
        self.assertFalse(r['projection']['observations'][0]['object_extent_known'])

    def test_empty_detection_cannot_hide_expiry_or_missing_context(self):
        for completed,now,context in [(.6,.6,True),(.3,.3,None),(.4,.3,True)]:
            with self.subTest(completed=completed,now=now,context=context):
                r=self.decision(context=context,completed=completed,now=now)
                self.assertFalse(r['permit_geometry_check'])
                self.assertEqual(r['reason'],'VISUAL_CONTEXT_HOLD')

    def test_frame_depth_registration_and_pose_time_are_checked(self):
        c=context_for(self.frame,'a',0)
        for bad in [replace(c,frame_id='other'),replace(c,registered_to_rgb=False),
                    replace(c,pose_at_s=.1),replace(c,depth_convention='range')]:
            with self.subTest(context=bad):
                self.assertFalse(self.decision(context=bad)['permit_geometry_check'])

    def test_veto_at_inclusive_age_boundary_and_bad_box(self):
        self.assertTrue(self.decision(completed=.5,now=.5)['permit_geometry_check'])
        self.assertFalse(self.decision(completed=.501,now=.501)['permit_geometry_check'])
        bad=dict(self.record,result=dict(boxes=[dict(group='person',score=.8,x1=-1,y1=1,x2=6,y2=5)]))
        with self.assertRaises(ValueError):self.decision(bad)

    def test_movement_gate_advances_latency_and_never_bypasses_geometry(self):
        frame=self.frame
        class Base:
            def move(self,target):
                self.geometry_calls+=1
                return dict(completed=False,reason='ORIGINAL_GEOMETRY_REFUSAL')
        class Vehicle(SemanticMoveMixin,Base):
            def state(self):return dict(time_s=self.time,velocity=[0,0,0])
            def capture(self,k,tick,aim):
                self.frames.append(frame)
                return frame
            def hold(self,seconds):self.time+=seconds
        class Detector:
            def detect(self,f):return dict(self_record,elapsed_s=.3001)
        self_record=self.record
        v=Vehicle();v.motor_latched_off=False;v.time=0.;v.semantic_checks=[]
        v.frames=[];v.saved=None;v.geometry_calls=0;v.detector=Detector()
        v.physics=type('Physics',(),dict(dt=.002))()
        v.spec=dict(semantic_delay_s=0.,semantic_wrong_frame=False)
        r=v.move((4,8))
        self.assertEqual(r['reason'],'ORIGINAL_GEOMETRY_REFUSAL')
        self.assertEqual(v.geometry_calls,1)
        self.assertAlmostEqual(v.time,.302)
        v.spec['semantic_delay_s']=.6
        r=v.move((4,8))
        self.assertEqual(r['reason'],'VISUAL_CONTEXT_HOLD')
        self.assertEqual(v.geometry_calls,1)
        self.assertFalse(r['completed'])
        self.assertEqual(v.semantic_checks[-1]['waiting_steps'],451)


if __name__=='__main__':unittest.main()
