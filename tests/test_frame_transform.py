"""来源：本项目原创。非平凡旋转、方向、单位、位姿与时效边界。"""
from copy import deepcopy
from dataclasses import FrozenInstanceError,replace
import unittest

from drone_nav.frame_transform import RigidFrameTransform,CameraPoseStamp,transform_delivery,finish_transform
from drone_nav.pinhole import Pose,Intrinsics
from drone_nav.timed_inbox import TimedObservationInbox


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.transform=RigidFrameTransform('source','anchor',((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.)),(10.,20.,30.),'fixture','a'*64)
        self.options=dict(source_frame='source',target_frame='anchor')
        self.pose=Pose((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
        self.k=Intrinsics(3,3,2,2,1,1)

    def delivery(self):
        source=dict(flight_authorized=False,navigation_frame_alignment_available=False,reason='VISUAL_TARGET_HOLD',
            projection=dict(frame_id='f',camera_id='c',clock_id='clock',consumer_clock_id='clock',detector_source='fixture',
                captured_at_s=1.,completed_at_s=1.1,consumer_at_s=1.2,age_s=.2,valid_until_s=1.5,
                world_frame='source',flight_authorized=False,free_space_evidence=[],status='SAMPLES_AVAILABLE',reasons=[],
                observations=[dict(association='unverified_box_surface',object_extent_known=False,
                    samples=[dict(pixel=[2,1],depth_z_m=2.,world_m=[1.,0.,2.])])]))
        inbox=TimedObservationInbox(camera_id='c',clock_id='clock',world_frame='source')
        inbox.submit(source,now_s=1.2,clock_id='clock')
        return inbox.consume(now_s=1.3,clock_id='clock',target_world='source')

    def convert(self,delivery=None,geometry=None,now=1.3,transform=None):
        return transform_delivery(delivery or self.delivery(),transform or self.transform,
            geometry or CameraPoseStamp('f','c','clock',1.,self.pose,self.k),now_s=now,clock_id='clock',target_frame='anchor')

    def test_rotation_translation_and_inverse_known_answer(self):
        point=self.transform.point((1,2,3),**self.options)
        self.assertEqual(point,(8.,21.,33.))
        self.assertEqual(self.transform.inverse().point(point,source_frame='anchor',target_frame='source'),(1.,2.,3.))
        self.assertEqual(self.transform.direction((1,2,3),**self.options),(-2.,1.,3.))

    def test_reference_pose_maps_anchor_to_identity(self):
        pose=Pose((3.,4.,5.),(0.,1.,0.),(-1.,0.,0.),(0.,0.,1.))
        transform=RigidFrameTransform.from_reference_pose(pose,source_frame='source',target_frame='anchor',reference_id='pose',reference_sha256='b'*64)
        target=transform.pose(pose,**self.options)
        self.assertEqual(target,self.pose)

    def test_rejects_scaling_reflection_skew_mutability_and_missing_reference(self):
        for r in (((2.,0.,0.),(0.,1.,0.),(0.,0.,1.)),((-1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),
                  ((1.,.1,0.),(0.,1.,0.),(0.,0.,1.)),[[1,0,0],[0,1,0],[0,0,1]]):
            with self.assertRaises(ValueError): replace(self.transform,rotation=r)
        for options in (dict(reference_sha256=''),dict(reference_id=''),dict(unit='cm'),dict(translation_m=[0,0,0]),dict(source_frame='anchor')):
            with self.assertRaises(ValueError): replace(self.transform,**options)
        with self.assertRaises(FrozenInstanceError): self.transform.translation_m=(0.,0.,0.)

    def test_direction_units_and_nonfinite_points_rejected(self):
        for options in (dict(source_frame='anchor',target_frame='source'),dict(source_frame='source',target_frame='anchor',unit='mm')):
            with self.assertRaises(ValueError): self.transform.point((1,2,3),**options)
        with self.assertRaises(ValueError): self.transform.point((1,2,float('nan')),**self.options)

    def test_transformed_pose_preserves_pixels_and_depth(self):
        delivery=self.delivery(); saved=deepcopy(delivery); converted=self.convert(delivery)
        self.assertEqual(delivery,saved)
        p=converted['camera_pose']; pose=Pose(*(tuple(p[k]) for k in ('position','right','down','forward')))
        sample=converted['observation']['projection']['observations'][0]['samples'][0]
        self.assertEqual(sample['world_m'],[10.,21.,32.])
        self.assertEqual(pose.project(sample['world_m'],self.k),(2.,1.,2.))
        self.assertFalse(converted['physical_navigation_calibrated'])

    def test_wrong_camera_pose_identity_and_timing_rejected(self):
        geometry=CameraPoseStamp('f','c','clock',1.,self.pose,self.k)
        for changes in (dict(frame_id='wrong'),dict(camera_id='wrong'),dict(pose_at_s=1.03),
                        dict(pose=replace(self.pose,position=(1.,0.,0.)))):
            with self.assertRaises(ValueError): self.convert(geometry=replace(geometry,**changes))

    def test_expiry_before_and_after_transform_and_no_authority(self):
        with self.assertRaises(ValueError): self.convert(now=1.501)
        converted=self.convert(now=1.49)
        at=finish_transform(converted,now_s=1.5,clock_id='clock')
        self.assertTrue(at['available']); self.assertFalse(at['navigation_map_update_allowed'])
        late=finish_transform(converted,now_s=1.5001,clock_id='clock')
        self.assertFalse(late['available']); self.assertIsNone(late['result'])
        with self.assertRaises(ValueError): finish_transform(converted,now_s=1.49,clock_id='wrong')

    def test_inverse_cannot_be_silently_used_as_forward(self):
        with self.assertRaises(ValueError): self.convert(transform=self.transform.inverse())


if __name__=='__main__': unittest.main()
