"""来源：本项目原创。新几何解除视图的边界和退化检查。"""
from copy import deepcopy
from dataclasses import replace
import unittest

from drone_nav.detection_bridge import DetectionPacket, SpatialContext
from drone_nav.pinhole import Intrinsics, Pose
from drone_nav.reobservation import (FOOTPRINT_MIN, ClearanceConfig, assess_clearance,
                                     coverage_window, reobserved_route)
from drone_nav.observation_channel import SurfaceEvidenceChannel
from drone_nav.occupancy import VoxelMap


def geometry(t=2):
    k = Intrinsics(32,32,10,10,15.5,15.5)
    p = DetectionPacket('new','depth-camera',32,32,(),'geometry',t,t+.05,'sim')
    c = SpatialContext('new','depth-camera','sim',t,t,k,Pose.look_at((0,2.5,1.5),(3,2.5,1.5)),
                       (10.,)*1024,True,'camera_optical_axis_z_m','ideal_footprint','exact_pose','k')
    return p,c


class ReobservationTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = dict(clock_id='sim', cells=[dict(voxel=[2,2,1],captured_at_s=1)])
        self.p,self.c = geometry()

    def assess(self, p=None, c=None, **kwargs):
        return assess_clearance(self.snapshot,p or self.p,c or self.c,now_s=kwargs.pop('now_s',2.1),
                                sampling_model=kwargs.pop('sampling_model',FOOTPRINT_MIN),**kwargs)['cells'][0]

    def test_clear_full_voxel_does_not_mutate_old_evidence(self):
        before=deepcopy(self.snapshot);r=self.assess()
        self.assertTrue(r['cleared']);self.assertEqual(self.snapshot,before)
        self.assertEqual(r['window'],[11,11,20,20]);self.assertEqual(r['checked_pixels'],100)
        self.assertAlmostEqual(r['far_z_m'],3.1)

    def test_point_depth_is_never_promoted_to_volume(self):
        self.assertIn('POINT_SAMPLES_CANNOT_CLEAR_VOLUME',self.assess(sampling_model='point_samples')['reasons'])

    def test_single_edge_pixel_missing_or_occluding_blocks(self):
        for z,reason in [(None,'MISSING_DEPTH'),(3.,'FOREGROUND_OR_INSUFFICIENT_CLEARANCE')]:
            d=list(self.c.depth_z_m);d[11*32+11]=z
            r=self.assess(c=replace(self.c,depth_z_m=tuple(d)))
            self.assertFalse(r['cleared']);self.assertIn(reason,r['reasons'])

    def test_equality_and_depth_error_not_relaxed(self):
        for d,clear in [(3.15,False),(3.15001,True)]:
            self.assertEqual(self.assess(c=replace(self.c,depth_z_m=(d,)*1024))['cleared'],clear)

    def test_view_and_behind_camera(self):
        self.assertIn('INCOMPLETE_VIEW',self.assess(c=replace(self.c,intrinsics=Intrinsics(32,32,100,100,15.5,15.5)))['reasons'])
        self.assertIn('NOT_FULLY_IN_FRONT',self.assess(c=replace(self.c,pose=Pose.look_at((0,2.5,1.5),(-1,2.5,1.5))))['reasons'])

    def test_age_is_rechecked_repeated_consumption_never_refreshes(self):
        self.assertTrue(self.assess(now_s=2.1)['cleared'])
        self.assertTrue(self.assess(now_s=2.5)['cleared'])
        self.assertIn('STALE_CAPTURE',self.assess(now_s=2.50001)['reasons'])

    def test_sync_interval_requires_strictly_newer_geometry(self):
        for t,clear in [(1.,False),(1.01,False),(1.02,False),(1.02001,True)]:
            p,c=geometry(t)
            self.assertEqual(self.assess(p,c,now_s=1.2)['cleared'],clear)
        self.snapshot['cells'][0]['captured_at_s']=2.1
        self.assertIn('NOT_NEWER_THAN_OLD_EVIDENCE',self.assess()['reasons'])

    def test_identity_timing_range_and_invalid_depth(self):
        for c,reason in [(replace(self.c,frame_id='wrong'),'FRAME_IDENTITY_MISMATCH'),
                         (replace(self.c,pose_at_s=2.05),'UNSYNCHRONIZED_CONTEXT'),
                         (replace(self.c,registered_to_rgb=False),'UNREGISTERED_DEPTH'),
                         (replace(self.c,depth_z_m=(100.,)*1024),'OUT_OF_RANGE')]:
            self.assertIn(reason,self.assess(c=c)['reasons'])
        with self.assertRaises(ValueError):self.assess(c=replace(self.c,depth_z_m=(float('nan'),)*1024))

    def test_bounds_and_corner_coverage(self):
        with self.assertRaises(ValueError):ClearanceConfig(spatial_margin_m=-1)
        with self.assertRaises(ValueError):ClearanceConfig(pixel_margin=True)
        with self.assertRaises(ValueError):coverage_window([2.5,2,1],self.c,ClearanceConfig())
        narrow=coverage_window([2,2,1],self.c,ClearanceConfig(0,0,0))
        self.assertLess(narrow['far_z_m'],self.assess()['far_z_m'])

    def test_route_restores_only_overlay_and_restricts_again_on_expiry(self):
        ch=SurfaceEvidenceChannel(6,6,4,clock_id='sim')
        p=DetectionPacket('old','old-camera',1,1,({'id':0,'group':'person','score':1,'box':[0,0,1,1]},),
                          'fixture',1,1.05,'sim')
        c=SpatialContext('old','old-camera','sim',1,1,Intrinsics(1,1,1,1,0,0),
                         Pose((2.5,2.5,.5),(1,0,0),(0,1,0),(0,0,1)),(1.,),True,
                         'camera_optical_axis_z_m','fixture','fixture','k')
        ch.ingest(p,c,now_s=1.1)
        grid=VoxelMap(6,6,4,ttl_ticks=100);grid.free_seen={(x,2,1):0 for x in range(1,5)}
        def route(now):
            return reobserved_route(grid,ch,self.p,self.c,sampling_model=FOOTPRINT_MIN,
                                    layer=1,tick=0,now_s=now,start=(1,2),goal=(4,2))
        r=route(2.1);self.assertEqual(r['before']['route'],[]);self.assertEqual(len(r['route']),4)
        grid.occupied.add((2,2,1));self.assertEqual(route(2.2)['route'],[])
        grid.occupied.clear();grid.free_seen={};self.assertEqual(route(2.3)['route'],[])
        grid.free_seen={(x,2,1):0 for x in range(1,5)}
        r=route(2.51);self.assertEqual(r['route'],[]);self.assertFalse(r['flight_authorized'])
        self.assertEqual(len(ch.snapshot(2.51)['cells']),1)

    def test_analytic_footprint_catches_subpixel_box_missed_by_centers(self):
        from tools.reobservation_experiment import footprint_depth
        k=Intrinsics(96,72,50,50,47.5,35.5)
        pose=Pose.look_at((10,8,1.5),(15,8,1.5))
        box=[14.51,7.991,1.21,14.52,7.993,1.22]
        depths=footprint_depth(k,pose,[box]);self.assertLess(min(depths),10)
        # 独立 slab 求交检查中心射线：该细盒没有中心射线命中。
        hits=0
        for v in range(k.height):
            for u in range(k.width):
                direction=pose.rotate(k.ray(u,v));lo=0;hi=float('inf')
                for axis in range(3):
                    origin=pose.position[axis];d=direction[axis]
                    if d==0:
                        if not box[axis]<=origin<=box[axis+3]:hi=-1
                    else:
                        a=(box[axis]-origin)/d;b=(box[axis+3]-origin)/d
                        lo=max(lo,min(a,b));hi=min(hi,max(a,b))
                hits+=lo<=hi
        self.assertEqual(hits,0)
        # 盒内点方向必定穿过盒；对应角域的下界不能晚于盒的入口。
        u,v,z=pose.project((14.515,7.992,1.215),k)
        self.assertLessEqual(depths[round(v)*96+round(u)],4.51+1e-9)

    def test_fixture_generator_lower_bound_under_rotated_view(self):
        from tools.reobservation_experiment import footprint_depth
        k=Intrinsics(32,24,20,20,15.5,11.5);pose=Pose.look_at((0,0,1),(5,3,1))
        box=[3,1,.2,4,2,1.8];depth=footprint_depth(k,pose,[box])
        # 采样盒的三维表面点；其像素角域的下界不大于该点光轴深度。
        for x in (3,3.2,3.5,3.8,4):
            for y in (1,1.3,1.6,2):
                for z in (.2,.7,1.2,1.8):
                    u,v,d=pose.project((x,y,z),k);px,py=round(u),round(v)
                    if 0<=px<32 and 0<=py<24:self.assertLessEqual(depth[py*32+px],d+1e-9)


if __name__=='__main__':unittest.main()
