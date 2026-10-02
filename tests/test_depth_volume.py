"""来源：本项目原创。深度体积检查的几何、缺测、坐标绑定与时效测试。"""
from copy import deepcopy
from dataclasses import asdict,replace
import unittest

from drone_nav.depth_volume import DepthVolumeInspector,QueryVolume
from drone_nav.pinhole import Intrinsics,Pose
from drone_nav.packed_rgbd import PackedObservation,PackedFrame
from drone_nav.frame_transform import RigidFrameTransform
from test_framed_surface_map import record

T=RigidFrameTransform('source','research',((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),(0.,0.,0.),'fixture','a'*64)
K=Intrinsics(5,5,4.,4.,2.,2.)
P=Pose((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.))


def inputs(z=4.,semantic=False):
    values=(z,)*25
    frame=PackedFrame(K,P,bytes(75),values)
    observation=PackedObservation(frame,'f1',1.,1.,1.,'fixture-registered-z')
    r=record(points=())
    r['result']['camera_pose']=asdict(P)
    if semantic:
        r=record(points=((0.,0.,z),)); r['result']['camera_pose']=asdict(P)
        sample=r['result']['observation']['projection']['observations'][0]['samples'][0]
        sample.update(pixel=[2,2],depth_z_m=z)
    r['result']['observation']['projection']['sources']={'depth':'fixture-registered-z'}
    return observation,r


class DepthVolumeTests(unittest.TestCase):
    def bind(self,obs,r): return DepthVolumeInspector(obs,r,T,camera_id='c',clock_id='clock',expected_intrinsics=K)
    def volume(self): return QueryVolume('centre','research',(-.2,-.2,1.),(.2,.2,2.))
    def inspect(self,view,volume=None,now=1.2): return view.inspect(volume or self.volume(),now_s=now,clock_id='clock')

    def test_far_samples_never_prove_free_volume(self):
        result=self.inspect(self.bind(*inputs()))
        self.assertEqual(result['status'],'SAMPLED_BEYOND_VOLUME')
        self.assertEqual(result['checked_pixels'],9)
        self.assertFalse(result['free_volume_proven']); self.assertFalse(result['flight_authorized'])

    def test_measured_surface_and_semantics_share_the_geometry(self):
        result=self.inspect(self.bind(*inputs(1.5,True)))
        self.assertEqual(result['status'],'MEASURED_SURFACE')
        self.assertEqual(result['inside_surface_pixels'],1)
        self.assertEqual(result['semantic_surface_samples'],1)
        self.assertEqual(result['witnesses'][0]['pixel'],[2,2])

    def test_occluder_is_not_a_hit_inside_the_box(self):
        result=self.inspect(self.bind(*inputs(.5)))
        self.assertEqual(result['status'],'FOREGROUND_OR_OCCLUSION')
        self.assertEqual(result['inside_surface_pixels'],0)

    def test_missing_and_range_are_unknown(self):
        for z,reason in ((None,'MISSING_DEPTH'),(50.,'OUT_OF_RANGE')):
            result=self.inspect(self.bind(*inputs(z)))
            self.assertEqual(result['status'],'INSUFFICIENT_DEPTH')
            self.assertIn(reason,result['reasons'])

    def test_outside_and_behind_camera_rejected_without_indexing(self):
        view=self.bind(*inputs())
        for lo,hi in (((2.,-.2,1.),(3.,.2,2.)),((-.2,-.2,-1.),(.2,.2,1.))):
            result=self.inspect(view,QueryVolume('bad','research',lo,hi))
            self.assertEqual(result['status'],'INSUFFICIENT_VIEW'); self.assertEqual(result['checked_pixels'],0)

    def test_earlier_depth_has_its_own_expiry(self):
        obs,r=inputs(); obs=replace(obs,depth_at_s=.99)
        view=self.bind(obs,r)
        self.assertNotEqual(self.inspect(view,now=1.49)['status'],'CONTEXT_REJECTED')
        self.assertEqual(self.inspect(view,now=1.491)['status'],'CONTEXT_REJECTED')

    def test_frame_time_pose_depth_and_reference_mismatch(self):
        for condition in ('frame','time','pose','depth','reference','calibration','source'):
            obs,r=inputs(1.5,True)
            if condition=='frame': obs=replace(obs,frame_id='other')
            elif condition=='time': obs=replace(obs,depth_at_s=.9)
            elif condition=='pose': r['result']['camera_pose']['position']=(.1,0.,0.)
            elif condition=='depth': obs=replace(obs,frame=replace(obs.frame,depth_z_m=(1.6,)*25))
            elif condition=='reference': r['result']['transform']['reference_sha256']='b'*64
            elif condition=='calibration': obs=replace(obs,frame=replace(obs.frame,intrinsics=replace(K,fx=5.)))
            elif condition=='source': obs=replace(obs,depth_source='unknown')
            with self.subTest(condition=condition),self.assertRaises(ValueError): self.bind(obs,r)

    def test_wrong_query_frame_clock_and_premature_consumer(self):
        view=self.bind(*inputs())
        with self.assertRaises(ValueError): self.inspect(view,replace(self.volume(),world_frame='ENU'))
        with self.assertRaises(ValueError): view.inspect(self.volume(),now_s=1.2,clock_id='wrong')
        with self.assertRaises(ValueError): self.inspect(view,now=1.19)

    def test_box_shape_and_nonfinite_margins(self):
        with self.assertRaises(ValueError): QueryVolume('x','research',(0.,0.,0.),(0.,1.,2.))
        with self.assertRaises(ValueError): QueryVolume('x','research',[0.,0.,0.],(1.,1.,2.))
        view=self.bind(*inputs())
        with self.assertRaises(ValueError): view.inspect(self.volume(),now_s=1.2,clock_id='clock',depth_error_m=float('nan'))

    def test_caller_mutation_does_not_change_bound_semantics(self):
        obs,r=inputs(1.5,True); before=deepcopy(r); view=self.bind(obs,r)
        self.assertEqual(r,before)
        r['result']['observation']['projection']['observations'][0]['samples'][0]['world_m'][0]=999
        self.assertEqual(self.inspect(view)['semantic_surface_samples'],1)


if __name__=='__main__': unittest.main()
