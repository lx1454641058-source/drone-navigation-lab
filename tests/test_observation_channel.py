"""来源：本项目原创。时效、重放隔离和 A* 附加限制的独立检查。"""
from dataclasses import replace
import unittest
from drone_nav.detection_bridge import DetectionPacket,SpatialContext
from drone_nav.pinhole import Intrinsics,Pose
from drone_nav.occupancy import VoxelMap
from drone_nav.observation_channel import SurfaceEvidenceChannel,shadow_route


def fixture(time=1,frame='first',camera='camera'):
    p=DetectionPacket(frame,camera,1,1,(dict(id=0,group='person',score=.9,box=[0,0,1,1]),),'fixture',time,time+.05,'sim')
    c=SpatialContext(frame,camera,'sim',time,time,Intrinsics(1,1,1,1,0,0),
                     Pose((2.5,2.5,.5),(1,0,0),(0,1,0),(0,0,1)),(1,),True,
                     'camera_optical_axis_z_m','controlled_depth','controlled_pose','k')
    return p,c


class ChannelTests(unittest.TestCase):
    def channel(self):return SurfaceEvidenceChannel(6,6,4,clock_id='sim')

    def test_duplicate_does_not_refresh_expiry(self):
        ch=self.channel();p,c=fixture();ch.ingest(p,c,now_s=1.1)
        self.assertEqual(ch.ingest(p,c,now_s=1.4)['reason'],'REPEATED_FRAME')
        s=ch.snapshot(1.51);self.assertEqual(s['cells'][0]['valid_until_s'],1.5)
        self.assertEqual(s['cells'][0]['state'],'unknown_after_expiry')

    def test_out_of_order_and_equal_capture_rejected(self):
        ch=self.channel();p,c=fixture();ch.ingest(p,c,now_s=1.1)
        for t in (.9,1):
            p,c=fixture(t,'other')
            self.assertEqual(ch.ingest(p,c,now_s=1.2)['reason'],'OUT_OF_ORDER_CAPTURE')

    def test_empty_new_frame_does_not_clear_old_cells(self):
        ch=self.channel();p,c=fixture();ch.ingest(p,c,now_s=1.1)
        p,c=fixture(1.2,'empty');r=ch.ingest(replace(p,detections=()),c,now_s=1.3)
        self.assertTrue(r['accepted']);self.assertEqual(r['updated_cells'],0)
        self.assertEqual(ch.snapshot(1.4)['cells'][0]['frame_id'],'first')

    def test_fresh_observation_restores_expired_cell(self):
        ch=self.channel();p,c=fixture();ch.ingest(p,c,now_s=1.1);ch.snapshot(1.51)
        p,c=fixture(1.6,'fresh');ch.ingest(p,c,now_s=1.7)
        s=ch.snapshot(1.8)['cells'][0];self.assertEqual(s['state'],'surface_restriction');self.assertEqual(s['valid_until_s'],2.1)

    def test_bad_batch_atomic_and_time_never_rewinds(self):
        ch=self.channel();p,c=fixture();ch.ingest(p,c,now_s=1.1);before=ch.snapshot(1.1)
        with self.assertRaises(ValueError):ch.ingest(p,replace(c,depth_z_m=(float('nan'),)),now_s=1.2)
        self.assertEqual(ch.snapshot(1.1),before)
        with self.assertRaises(ValueError):ch.snapshot(1)

    def test_rejected_context_and_old_other_camera_do_not_refresh(self):
        ch=self.channel();p,c=fixture();ch.ingest(p,c,now_s=1.1)
        self.assertFalse(ch.ingest(p,replace(c,registered_to_rgb=False),now_s=1.2)['accepted'])
        p,c=fixture(.95,'other','camera-2');r=ch.ingest(p,c,now_s=1.3)
        self.assertEqual(r['updated_cells'],0);self.assertEqual(ch.snapshot(1.4)['cells'][0]['captured_at_s'],1)

    def test_boundary_protects_both_sides_and_snapshot_is_detached(self):
        ch=self.channel();self.assertEqual(set(ch._voxels((2,2,1))),{(x,y,z) for x in (1,2) for y in (1,2) for z in (0,1)})
        p,c=fixture();ch.ingest(p,c,now_s=1.1);s=ch.snapshot(1.1);s['cells'][0]['references'].clear()
        self.assertTrue(ch.snapshot(1.1)['cells'][0]['references'])

    def test_route_detours_and_expiry_does_not_reopen(self):
        ch=self.channel();p,c=fixture();ch.ingest(p,c,now_s=1.1)
        g=VoxelMap(6,6,4,ttl_ticks=100)
        g.free_seen={(x,y,1):0 for x in range(1,5) for y in range(1,4)}
        r=shadow_route(g,ch,layer=1,tick=0,now_s=1.2,start=(1,2),goal=(4,2))
        self.assertIn((2,2),r['baseline_route']);self.assertNotIn((2,2),r['route']);self.assertGreater(len(r['route']),len(r['baseline_route']))
        expired=shadow_route(g,ch,layer=1,tick=0,now_s=1.6,start=(1,2),goal=(4,2))
        self.assertEqual(expired['route'],r['route']);self.assertFalse(expired['flight_authorized'])

    def test_narrow_and_unknown_corridors_hold(self):
        ch=self.channel();p,c=fixture();ch.ingest(p,c,now_s=1.1)
        for cells in ({(x,2,1):0 for x in range(1,5)},{}):
            g=VoxelMap(6,6,4);g.free_seen=cells
            r=shadow_route(g,ch,layer=1,tick=0,now_s=1.2,start=(1,2),goal=(4,2))
            self.assertEqual(r['status'],'HOLD');self.assertEqual(r['route'],[])


if __name__=='__main__':unittest.main()
