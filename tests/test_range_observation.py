"""来源：本项目原创。未命中与失效、径向距离及原拒绝行为的边界测试。"""
from dataclasses import replace
import unittest

from drone_nav.pinhole import Intrinsics, Pose, reconstruct
from drone_nav.raycast import World, Box
from drone_nav.range_observation import (render_range, legacy_frame, RangeBoxInspector,
                                         RangeSamplingHistory)
from drone_nav.sampling_motion import SamplingView, SamplingAssumption
from drone_nav.physical_vehicle import FlightBudget, spatial_cells
from drone_nav.refined_sampling import InitialClearRegion


class RangeObservationTests(unittest.TestCase):
    def setUp(self):
        self.k = Intrinsics(40,30,25,25,19.5,14.5)
        self.pose = Pose.look_at((0,0,0),(0,0,1))
        self.a = SamplingAssumption(.2,.2)
        self.box = ((-.1,-.1,2.),(.1,.1,2.25))

    def frame(self, **kw):
        return render_range(World(()), self.pose, self.k, **kw)

    def inspect(self, f, box=None, assumption=True):
        v = SamplingView(f,'f','camera','physics-seconds',0.,0.)
        return RangeBoxInspector(v,self.a if assumption else None).inspect(*(box or self.box))

    def test_healthy_miss_and_sensor_fault_are_distinct(self):
        healthy, broken = self.frame(), self.frame(invalid=True)
        self.assertEqual(healthy.depth_z_m,broken.depth_z_m)
        self.assertEqual(set(healthy.outcomes),{'NO_HIT'})
        self.assertEqual(set(broken.outcomes),{'INVALID'})
        self.assertIsNone(self.inspect(healthy))
        self.assertEqual(self.inspect(broken),'SENSOR_INVALID')

    def test_no_miss_becomes_fake_surface_in_existing_reconstruction(self):
        f = self.frame()
        self.assertEqual(reconstruct(f),[])
        self.assertEqual(reconstruct(legacy_frame(f)),[])
        with self.assertRaises(ValueError):self.inspect(legacy_frame(f))

    def test_actual_hit_is_preserved_and_foreground_blocks(self):
        world=World((Box('block',3,(-1,-1,1.5),(1,1,1.7)),))
        f=render_range(world,self.pose,self.k)
        self.assertEqual(f.outcomes[15*40+20],'HIT')
        self.assertEqual(f.depth_z_m[15*40+20],1.5)
        self.assertEqual(self.inspect(f),'RANGE_BOUND_OR_FOREGROUND')

    def test_range_is_radial_and_has_strict_margin(self):
        # 中心 z=2.25 可在 2.5 米内，偏轴角落的射线路程却超过该量程。
        f=self.frame(max_range_m=2.5)
        self.assertIsNone(self.inspect(f))
        self.assertEqual(self.inspect(f,((1.2,-.1,2.),(1.3,.1,2.25))),
                         'RANGE_BOUND_OR_FOREGROUND')
        self.assertEqual(self.inspect(self.frame(max_range_m=2.27)),
                         'RANGE_BOUND_OR_FOREGROUND')

    def test_feature_size_and_full_projection_are_still_required(self):
        self.assertEqual(self.inspect(self.frame(),assumption=False),'MINIMUM_FEATURE_ASSUMPTION_UNKNOWN')
        self.assertEqual(self.inspect(self.frame(),((-.1,)*3,(.1,)*3)),'NOT_FULLY_IN_FRONT')
        v=SamplingView(self.frame(),'f','camera','physics-seconds',0.,0.)
        self.assertEqual(RangeBoxInspector(v,SamplingAssumption(.001,.001)).inspect(*self.box),'SAMPLING_TOO_COARSE')

    def test_contradictory_outcomes_and_bad_ranges_are_rejected(self):
        f=self.frame()
        for bad in (replace(f,outcomes=('HIT',)*1200),replace(f,outcomes=('UNKNOWN',)*1200),
                    replace(f,max_range_m=True),replace(f,max_range_m=float('inf')),
                    replace(f,max_range_m=31.),replace(f,sensor_model='real-camera')):
            with self.subTest(bad=bad.sensor_model):
                with self.assertRaises(ValueError):bad.validate()
        with self.assertRaises(ValueError):self.frame(invalid=1)

    def test_nohit_does_not_populate_original_map_or_clear_occupied(self):
        h=RangeSamplingHistory()
        f=render_range(World(()),Pose.look_at((3.5,8.5,3.5),(4.5,8.5,3.5)),self.k)
        h.grid.occupied.add((4,8,3))
        h.add(SamplingView(f,'f','camera','physics-seconds',0.,0.),now_s=0.)
        self.assertFalse(h.grid.free_seen)
        self.assertIn((4,8,3),h.grid.occupied)
        q=h.check((3,8),(4,8),now_s=0.,budget=FlightBudget(),assumption=self.a)
        self.assertFalse(q['allowed'])
        self.assertFalse(next(r for r in q['sampling'] if r['voxel']==[4,8,3])['supported'])

    def test_expired_range_frames_cannot_fill_patches(self):
        h=RangeSamplingHistory()
        f=render_range(World(()),Pose.look_at((3.5,8.5,3.5),(4.5,8.5,3.5)),self.k)
        h.add(SamplingView(f,'f','camera','physics-seconds',0.,0.),now_s=0.)
        # 隔离时间检查：地图来源很新，唯一图像到停止时却已过期。
        h.stamps[1]=4.;h.grid.last_tick=1
        for c in spatial_cells((3,8),(4,8),.77):h.grid.free_seen[c]=1
        q=h.check((3,8),(4,8),now_s=4.,budget=FlightBudget(),assumption=self.a)
        self.assertTrue(q['baseline']['allowed'])
        self.assertFalse(q['allowed'])
        self.assertTrue(all(p['source'] is None for r in q['sampling'] for p in r['patches']))


if __name__=='__main__':unittest.main()
