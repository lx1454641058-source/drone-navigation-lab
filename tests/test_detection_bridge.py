"""来源：本项目原创。独立解析坐标、时间错配及不产生空闲证据的检查。"""
from dataclasses import replace
import unittest
from drone_nav.detection_bridge import DetectionPacket,SpatialContext,BridgeConfig,project_detections
from drone_nav.pinhole import Intrinsics,Pose


def fixture():
    p=DetectionPacket('frame','camera',3,3,(dict(id=0,group='person',score=.8,box=[0,0,3,3]),),'fixture',1,1.1,'sim')
    c=SpatialContext('frame','camera','sim',1,1,Intrinsics(3,3,1,1,1,1),Pose((10,20,30),(1,0,0),(0,1,0),(0,0,1)),(2,)*9,True,'camera_optical_axis_z_m','sim_depth','sim_pose','test_k')
    return p,c


def run(p,c,**kwargs):
    return project_detections(p,c,now_s=kwargs.pop('now_s',1.2),now_clock_id=kwargs.pop('now_clock_id','sim'),**kwargs)


class BridgeTests(unittest.TestCase):
    def test_analytic_world_coordinates_and_no_clearance_claim(self):
        p,c=fixture();r=run(p,c);points=r['observations'][0]['samples']
        self.assertEqual(points[0]['world_m'],[8,18,32]);self.assertEqual(points[4]['world_m'],[10,20,32])
        self.assertEqual(points[-1]['world_m'],[12,22,32]);self.assertEqual(r['valid_until_s'],1.5)
        self.assertEqual(r['free_space_evidence'],[]);self.assertFalse(r['flight_authorized'])
        self.assertFalse(r['observations'][0]['object_extent_known'])

    def test_rotated_camera_uses_capture_pose(self):
        p,c=fixture();pose=Pose((10,20,30),(0,1,0),(0,0,-1),(-1,0,0))
        r=run(p,replace(c,pose=pose));points=r['observations'][0]['samples']
        self.assertEqual(points[4]['world_m'],[8,20,30]);self.assertEqual(points[0]['world_m'],[8,18,32])

    def test_capture_age_not_completion_age(self):
        p,c=fixture();r=run(replace(p,completed_at_s=1.7),c,now_s=1.71)
        self.assertIn('STALE_CAPTURE',r['reasons']);self.assertEqual(r['observations'][0]['samples'],[])
        self.assertEqual(run(p,c,now_s=1.5)['status'],'SAMPLES_AVAILABLE')
        self.assertIn('STALE_CAPTURE',run(p,c,now_s=1.5001)['reasons'])

    def test_time_order_future_and_missing_timestamps(self):
        p,c=fixture()
        for new in (replace(p,completed_at_s=.9),replace(p,completed_at_s=2)):
            self.assertIn('INVALID_TIME_ORDER',run(new,c)['reasons'])
        r=run(replace(p,captured_at_s=None),c)
        self.assertIn('MISSING_TIMING',r['reasons']);self.assertIsNone(r['valid_until_s'])
        self.assertIn('FUTURE_CONTEXT',run(p,replace(c,pose_at_s=2))['reasons'])

    def test_identity_clock_size_registration_and_depth_convention(self):
        p,c=fixture()
        for new,reason in [(replace(c,frame_id='other'),'FRAME_IDENTITY_MISMATCH'),(replace(c,camera_id='other'),'FRAME_IDENTITY_MISMATCH'),
                           (replace(c,clock_id='other'),'CLOCK_MISMATCH'),(replace(c,registered_to_rgb=False),'UNREGISTERED_DEPTH'),
                           (replace(c,depth_convention='ray_range_m'),'DEPTH_CONVENTION_MISMATCH'),
                           (replace(c,intrinsics=Intrinsics(1,1,1,1,0,0),depth_z_m=(2,)),'IMAGE_SIZE_MISMATCH')]:
            r=run(p,new);self.assertIn(reason,r['reasons']);self.assertFalse(r['observations'][0]['samples'])
        r=run(p,c,now_clock_id='other');self.assertIn('CONSUMER_CLOCK_MISMATCH',r['reasons']);self.assertIsNone(r['age_s'])

    def test_sync_and_missing_context(self):
        p,c=fixture()
        self.assertEqual(run(p,replace(c,depth_at_s=1.02))['status'],'SAMPLES_AVAILABLE')
        self.assertIn('UNSYNCHRONIZED_CONTEXT',run(p,replace(c,depth_at_s=1.020001))['reasons'])
        self.assertIn('UNSYNCHRONIZED_CONTEXT',run(p,replace(c,depth_at_s=1.1))['reasons'])
        self.assertIn('UNSYNCHRONIZED_CONTEXT',run(p,replace(c,pose_at_s=.8))['reasons'])
        self.assertIn('MISSING_SPATIAL_CONTEXT',run(p,None)['reasons'])

    def test_missing_depth_and_partial_support(self):
        p,c=fixture();r=run(p,replace(c,depth_z_m=(None,)*9))
        self.assertEqual(r['status'],'NO_SPATIAL_SAMPLES');self.assertEqual(r['observations'][0]['missing_depth'],9)
        r=run(p,replace(c,depth_z_m=(None,)*4+(2,)+(None,)*4))
        self.assertEqual(len(r['observations'][0]['samples']),1);self.assertIn('PARTIAL_DEPTH_SUPPORT',r['observations'][0]['reasons'])

    def test_range_is_slant_not_axial(self):
        p,c=fixture();r=run(p,c,config=BridgeConfig(max_range_m=2.1))
        self.assertEqual(len(r['observations'][0]['samples']),1);self.assertEqual(r['observations'][0]['out_of_range'],8)

    def test_half_open_windows_and_empty_detection_no_free_space(self):
        p,c=fixture();d=dict(p.detections[0],box=[.1,.1,.9,.9])
        r=run(replace(p,detections=(d,)),c)
        self.assertEqual(r['observations'][0]['reasons'],['NO_PIXEL_CENTERS'])
        r=run(replace(p,detections=()),c)
        self.assertEqual(r['status'],'NO_SPATIAL_SAMPLES');self.assertEqual(r['free_space_evidence'],[])

    def test_invalid_data_rejected_before_projection(self):
        p,c=fixture()
        for new in (replace(p,width=True),replace(p,captured_at_s=float('nan')),replace(p,detections=p.detections*2)):
            with self.assertRaises(ValueError):run(new,c)
        for depths in ((2,),(float('nan'),)*9,(-1,)*9,(True,)*9):
            with self.assertRaises(ValueError):run(p,replace(c,depth_z_m=depths))
        with self.assertRaises(ValueError):BridgeConfig(samples_per_axis=1)


if __name__=='__main__':unittest.main()
