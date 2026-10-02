"""来源：本项目原创。实际中心射线与采样统计的反例验证。"""
from copy import deepcopy
import unittest
from drone_nav.sampling_coverage import coverage_scenes,camera,scene_world,depth_candidates,front_projection,summarize
from drone_nav.raycast import first_hit,render


def ray_hits(scene,width,view):
    k,pose=camera(width,view);world=scene_world(scene)
    return sum((hit:=first_hit(world,pose.position,pose.rotate(k.ray(u,v)))) is not None and hit[1].name=='target'
               for v in range(k.height) for u in range(k.width))


class SamplingCoverageTests(unittest.TestCase):
    def test_matrix_preserves_world_and_camera_field_of_view(self):
        scenes=coverage_scenes();self.assertEqual(len(scenes),96);self.assertEqual(len({s['key'] for s in scenes}),96)
        for width in (48,96,192):
            k,p=camera(width,0)
            self.assertEqual(k.width/k.fx,96/50);self.assertEqual(k.height/k.fy,72/50)
            self.assertEqual(p.position,(0,8,4))
        with self.assertRaises(ValueError):camera(96,True)

    def test_resolution_increase_does_not_preserve_each_center_ray(self):
        s=next(s for s in coverage_scenes() if s['distance_m']==3 and s['size_m']==.005 and s['phase']==[.5,.5])
        self.assertGreater(ray_hits(s,96,0),0)
        self.assertEqual(ray_hits(s,192,0),0)

    def test_large_front_face_contains_center_sample(self):
        for s in coverage_scenes():
            if s['size_m']!=.1 or s['distance_m']!=3:continue
            k,p=camera(96,0)
            self.assertTrue(front_projection(s,k,p)['front_face_sampling_condition'])
            self.assertGreater(ray_hits(s,96,0),0)

    def test_real_rgbd_candidates_equal_separate_truth_mask(self):
        s=next(s for s in coverage_scenes() if s['size_m']==.1 and s['distance_m']==3)
        k,p=camera(48,0);f,t=render(scene_world(s),p,k)
        self.assertEqual(depth_candidates(f.depth_z_m),[i for i,n in enumerate(t.objects) if n=='target'])
        self.assertEqual(depth_candidates([None,16,15,3]),[3])

    def test_invalid_depth_not_silently_ignored(self):
        for z in (0,-1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):depth_candidates([z])
        with self.assertRaises(ValueError):depth_candidates([1],nearer_than_m=True)

    def test_multiview_union_not_sum_of_detected_positions(self):
        rows=[dict(scene_key='a',width=96,height=72,view=i,distance_m=3,size_m=.1,target_pixels=n)
              for i,n in enumerate((0,1,2))]
        r=summarize(rows)[0]
        self.assertEqual((r['single_hits'],r['three_view_hits'],r['positions']),(0,1,1))
        self.assertEqual(r['three_view_target_pixels'],3)
        self.assertEqual(r['three_view_sample_count'],3*96*72)

    def test_incomplete_duplicate_or_mixed_group_rejected(self):
        rows=[dict(scene_key='a',width=96,height=72,view=i,distance_m=3,size_m=.1,target_pixels=0) for i in range(3)]
        with self.assertRaises(ValueError):summarize(rows[:2])
        with self.assertRaises(ValueError):summarize(rows+[rows[0]])
        other=deepcopy(rows);other[2]['size_m']=.2
        with self.assertRaises(ValueError):summarize(other)


if __name__=='__main__':unittest.main()
