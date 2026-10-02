"""来源：本项目原创。真实 RGB-D 输入的单位、坐标与时序边界。"""
import tempfile
import struct
import unittest
from dataclasses import replace
from math import sqrt
from pathlib import Path

from drone_nav.pinhole import Pose
from drone_nav.tum_rgbd import (CLOCK, WORLD, INTRINSICS, RGBDObservation,
    load_observation, nearest, pose_from_record, project_record, rows, safe_path)


class TUMInputTests(unittest.TestCase):
    def test_nearest_boundary_and_tie(self):
        self.assertEqual(nearest([(1.0, 'a'), (1.04, 'b')], 1.02), (1., 'a'))
        with self.assertRaises(ValueError): nearest([(1., 'a')], 1.0201)
        with self.assertRaises(ValueError): nearest([], 1.)
        with self.assertRaises(ValueError): nearest([(1., 'a')], float('nan'))

    def test_records_require_order_and_finite_time(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'index.txt'
            p.write_text('# sample\n1 rgb/1.png\n2 rgb/2.png\n')
            self.assertEqual(rows(p, 2)[1], (2., 'rgb/2.png'))
            for text in ('2 x\n1 y', '1 x\n1 y', 'nan x', '1 x y', ''):
                p.write_text(text)
                with self.assertRaises(ValueError): rows(p, 2)

    def test_dataset_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(safe_path(d, 'rgb/1.png'), Path(d).resolve()/'rgb/1.png')
            for path in ('../x', '/x', 'C:/x', 'rgb\\x', ''):
                with self.assertRaises(ValueError): safe_path(d, path)

    def test_quaternion_axes_and_translation(self):
        q = sqrt(.5)
        pose = pose_from_record((1., '1', '2', '3', '0', '0', str(q), str(q)))
        # 90 degrees around world Z: camera right -> +Y, down -> -X.
        point = pose.unproject(INTRINSICS.cx+INTRINSICS.fx, INTRINSICS.cy, 2., INTRINSICS)
        for got, want in zip(point, (1., 4., 5.)): self.assertAlmostEqual(got, want)
        for quaternion in (('0','0','0','0'), ('0','0','0','2'), ('nan','0','0','1')):
            with self.assertRaises(ValueError): pose_from_record((1., '0','0','0', *quaternion))

    def test_real_png_depth_units_missing_and_context(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            Image.new('RGB', (640,480), (30,40,50)).save(Path(d)/'rgb.png')
            depth = Image.new('I;16', (640,480), 5000)
            depth.putpixel((0,0), 0)
            depth.save(Path(d)/'depth.png')
            selection = dict(rgb=(1.,'rgb.png'), depth=(1.01,'depth.png'),
                pose=(.995,'1','2','3','0','0','0','1'))
            observation = load_observation(d, selection, .9)
            self.assertIsNone(observation.frame.depth_z_m[0])
            self.assertEqual(observation.frame.depth_z_m[1], 1.)
            self.assertAlmostEqual(observation.depth_at_s-observation.captured_at_s, .01)
            record = dict(model_sha256='test-model', result=dict(boxes=[
                dict(group='person',score=.8,x1=300,y1=230,x2=330,y2=250)]))
            result = project_record(observation, record, completed_at_s=.12, now_s=.12)
            self.assertEqual(result['reason'], 'VISUAL_TARGET_HOLD')
            self.assertEqual(result['projection']['world_frame'], WORLD)
            self.assertFalse(result['flight_authorized'])
            self.assertFalse(result['navigation_frame_alignment_available'])
            self.assertEqual(len(result['projection']['observations'][0]['samples']), 25)
            self.assertEqual(project_record(observation, record, completed_at_s=.7, now_s=.7)['reason'], 'VISUAL_CONTEXT_HOLD')
            record['result']['boxes'] = []
            result = project_record(observation, record, completed_at_s=.12, now_s=.12)
            self.assertEqual(result['reason'], 'NO_DETECTION_REQUIRES_GEOMETRY')
            self.assertEqual(result['projection']['free_space_evidence'], [])
            with self.assertRaises(ValueError):
                load_observation(d, dict(selection, depth=(1.021,'depth.png')), .9)

    def test_bag_float_metres_are_not_rescaled(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            Image.new('RGB',(640,480)).save(Path(d)/'rgb.png')
            p=Path(d)/'depth.float32'
            p.write_bytes(struct.pack('<3f',float('nan'),0,1.2345)+struct.pack('<f',2.)*(640*480-3))
            selected=dict(rgb=(1.,'rgb.png'),depth=(1.,p.name),pose=(1.,'0','0','0','0','0','0','1'),depth_format='32FC1_LE')
            observation=load_observation(d,selected,0.)
            self.assertEqual(observation.frame.depth_z_m[:2],(None,None))
            self.assertAlmostEqual(observation.frame.depth_z_m[2],1.2345,places=6)
            self.assertEqual(observation.frame.depth_z_m[3],2.)
            p.write_bytes(struct.pack('<f',-1.)*(640*480))
            with self.assertRaises(ValueError): load_observation(d,selected,0.)


if __name__ == '__main__':
    unittest.main()
