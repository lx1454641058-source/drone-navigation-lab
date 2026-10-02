"""来源：本项目原创。细分覆盖不缩小范围、外部假设时效与观测优先级。"""
from dataclasses import replace
from itertools import product
import unittest

from drone_nav.refined_sampling import InitialClearRegion, RefinedSamplingHistory, inspect_box
from drone_nav.physical_vehicle import FlightBudget, spatial_cells
from drone_nav.pinhole import Pose, Intrinsics, PerspectiveFrame
from drone_nav.sampling_motion import SamplingView, SamplingAssumption, inspect_voxel


class RefinedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.a=SamplingAssumption(.2,.2)
        self.seed=InitialClearRegion((2.,7.,3.),(6.,10.,4.),0.,12.,'test-explicit')

    def history(self,seed=None,n=4):
        h=RefinedSamplingHistory(divisions=n,initial_region=seed)
        h.grid.last_tick=0;h.stamps[0]=0.
        for c in spatial_cells((3,8),(4,8),.77):h.grid.free_seen[c]=0
        return h

    def test_partition_covers_exact_original_cells_without_overlap(self):
        h=self.history(self.seed)
        q=h.check((3,8),(4,8),now_s=1.,budget=FlightBudget(),assumption=self.a)
        self.assertTrue(q['allowed'])
        self.assertEqual({tuple(c) for c in q['required_cells']},spatial_cells((3,8),(4,8),.77))
        for row in q['sampling']:
            self.assertEqual(len(row['patches']),64)
            expected={tuple(c+i/4 for c,i in zip(row['voxel'],ix)) for ix in product(range(4),repeat=3)}
            self.assertEqual({tuple(p['low']) for p in row['patches']},expected)
            self.assertEqual(sum((p['high'][0]-p['low'][0])*(p['high'][1]-p['low'][1])*(p['high'][2]-p['low'][2]) for p in row['patches']),1.)

    def test_seed_does_not_override_observed_obstacle_or_baseline(self):
        h=self.history(self.seed);h.grid.occupied.add((3,8,3))
        q=h.check((3,8),(4,8),now_s=1.,budget=FlightBudget(),assumption=self.a)
        self.assertFalse(q['allowed'])
        row=next(r for r in q['sampling'] if r['voxel']==[3,8,3])
        self.assertTrue(all(p['source'] is None and 'OBSERVED_OCCUPIED' in p['reasons'] for p in row['patches']))
        h=self.history(self.seed);h.grid.free_seen.clear()
        self.assertFalse(h.check((3,8),(4,8),now_s=1.,budget=FlightBudget(),assumption=self.a)['allowed'])

    def test_unknown_expired_future_or_wrong_clock_seed_cannot_fill_space(self):
        for seed in (None,replace(self.seed,valid_until_s=9.),replace(self.seed,captured_at_s=2.),
                     replace(self.seed,clock_id='other-clock')):
            with self.subTest(seed=seed):
                q=self.history(seed).check((3,8),(4,8),now_s=1.,budget=FlightBudget(),assumption=self.a)
                self.assertFalse(q['allowed']);self.assertFalse(q['initial_region_active'])
        self.assertFalse(self.history(self.seed).check((3,8),(4,8),now_s=1.,budget=FlightBudget())['allowed'])

    def test_box_must_be_fully_contained_by_seed(self):
        h=self.history(InitialClearRegion((3.01,8.,3.),(4.,9.,4.),0.,12.,'partial'))
        q=h.check((3,8),(4,8),now_s=1.,budget=FlightBudget(),assumption=self.a)
        row=next(r for r in q['sampling'] if r['voxel']==[3,8,3])
        self.assertEqual(sum(p['source'] is not None for p in row['patches']),48)
        self.assertFalse(row['supported'])

    def view(self,depth=10.,pose=None):
        k=Intrinsics(40,30,25,25,19.5,14.5)
        pose=pose or Pose.look_at((0.,0.,0.),(0.,0.,1.))
        f=PerspectiveFrame(k,pose,((0,0,0),)*1200,(depth,)*1200,0)
        return SamplingView(f,'f','camera','physics-seconds',0.,0.)

    def test_full_voxel_agrees_with_original_geometry_conditions(self):
        for cell in ((0,0,2),(-1,-1,2),(0,0,0),(7,7,2)):
            v=self.view()
            self.assertEqual(inspect_box(cell,tuple(x+1 for x in cell),v,self.a) is None,
                             not inspect_voxel(cell,v,self.a)['reasons'])

    def test_missing_depth_foreground_and_fine_size_remain_rejected(self):
        low=(-.1,-.1,2.);high=(.1,.1,2.25)
        self.assertIsNone(inspect_box(low,high,self.view(),self.a))
        self.assertEqual(inspect_box(low,high,self.view(None),self.a),'MISSING_DEPTH')
        self.assertEqual(inspect_box(low,high,self.view(2.25),self.a),'FOREGROUND_OR_DEPTH_MARGIN')
        self.assertEqual(inspect_box(low,high,self.view(),SamplingAssumption(.001,.001)),'SAMPLING_TOO_COARSE')

    def test_no_camera_can_fully_project_box_containing_its_origin(self):
        for direction in ((1,0,0),(0,1,0),(0,0,1),(-1,-1,-1)):
            v=self.view(pose=Pose.look_at((0.,0.,0.),direction))
            self.assertEqual(inspect_box((-.25,)*3,(.25,)*3,v,self.a),'NOT_FULLY_IN_FRONT')

    def test_invalid_configuration_rejected(self):
        for n in (0,5,True,1.5):
            with self.assertRaises(ValueError):RefinedSamplingHistory(divisions=n)
        with self.assertRaises(ValueError):InitialClearRegion((0,0,0),(1,1,1),0.,float('nan'),'s')
        with self.assertRaises(ValueError):InitialClearRegion((0,0,0),(1,1,1),0.,1.,'')


if __name__=='__main__':unittest.main()
