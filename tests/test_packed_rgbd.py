"""来源：本项目原创。输入优化必须保持像素、舍入、缺失值和时间契约。"""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from random import Random
import struct
import tempfile
import unittest

from PIL import Image
from drone_nav.pinhole import Intrinsics,Pose
from drone_nav.packed_rgbd import PackedFrame,load_packed,prepare_packed,handoff_decision
from drone_nav.tum_rgbd import load_observation
from tools.tinyformer_probe import prepare


class PackedInputTests(unittest.TestCase):
    def test_projection_processing_cannot_use_expired_empty_frame(self):
        result=dict(reason='NO_DETECTION_REQUIRES_GEOMETRY',projection=dict(
            consumer_at_s=.49,valid_until_s=.5,consumer_clock_id='test'))
        self.assertTrue(handoff_decision(result,consumer_at_s=.5,clock_id='test')['permit_geometry_check'])
        late=handoff_decision(result,consumer_at_s=.501,clock_id='test')
        self.assertFalse(late['permit_geometry_check']); self.assertTrue(late['expired_at_handoff'])
        result['reason']='VISUAL_TARGET_HOLD'
        self.assertFalse(handoff_decision(result,consumer_at_s=.5,clock_id='test')['permit_geometry_check'])
        for now,clock in ((.48,'test'),(.5,'wrong'),(float('nan'),'test')):
            with self.assertRaises(ValueError): handoff_decision(result,consumer_at_s=now,clock_id=clock)

    def test_all_byte_values_and_non_square_crop_match_reference(self):
        image=Image.new('RGB',(256,3))
        image.putdata([(x,(x+79)%256,255-x) for y in range(3) for x in range(256)])
        for window in (dict(x0=0,y0=0,width=256,height=3),dict(x0=9,y0=1,width=133,height=2)):
            self.assertEqual(prepare_packed(image,window),prepare(image,window))

    def test_seeded_rgba_conversion_and_resize_are_identical(self):
        rng=Random(217)
        image=Image.frombytes('RGBA',(37,23),bytes(rng.randrange(256) for _ in range(37*23*4)))
        window=dict(x0=1,y0=2,width=31,height=17)
        self.assertEqual(prepare_packed(image,window),prepare(image,window))
        for bad in (dict(x0=-1,y0=0,width=4,height=3),dict(x0=0,y0=0,width=38,height=3),
                    dict(x0=True,y0=0,width=3,height=3),dict(x0=0,y0=0,width=0,height=3)):
            with self.assertRaises(ValueError): prepare_packed(image,bad)

    def test_immutable_frame_rejects_mutable_or_malformed_input(self):
        k=Intrinsics(1,1,1,1,0,0); p=Pose((0,0,0),(1,0,0),(0,1,0),(0,0,1))
        frame=PackedFrame(k,p,b'\0\xff\x80',(2.,))
        with self.assertRaises(FrozenInstanceError): frame.rgb_bytes=b'abc'
        for rgb in (bytearray(b'abc'),b'ab',[(1,2,3)]):
            with self.assertRaises(ValueError): PackedFrame(k,p,rgb,(2.,))
        for depth in ([2.],(),(0.,),(float('nan'),),(float('inf'),),(-1.,),(True,)):
            with self.assertRaises(ValueError): PackedFrame(k,p,b'abc',depth)

    def test_loader_keeps_depth_pose_pixels_and_rejects_invalid_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); image=Image.new('RGB',(640,480),(19,111,255)); image.save(root/'rgb.png')
            depth=root/'depth.float32'
            depth.write_bytes(struct.pack('<3f',0,float('nan'),1.23)+struct.pack('<f',2.)*(640*480-3))
            selection=dict(rgb=(1.,'rgb.png'),depth=(1.01,depth.name),pose=(1.,'1','2','3','0','0','0','1'),depth_format='32FC1_LE')
            packed=load_packed(root,selection,.9); old=load_observation(root,selection,.9)
            self.assertEqual(packed.frame.rgb_bytes,image.tobytes())
            self.assertEqual(tuple(zip(*[iter(packed.frame.rgb_bytes)]*3)),old.frame.rgb)
            for attr in ('depth_z_m','pose','intrinsics'):
                self.assertEqual(getattr(packed.frame,attr),getattr(old.frame,attr))
            for attr in ('captured_at_s','depth_at_s','pose_at_s','depth_source'):
                self.assertEqual(getattr(packed,attr),getattr(old,attr))
            with self.assertRaises(ValueError): load_packed(root,dict(selection,depth=(1.021,depth.name)),.9)
            for value in (-1.,float('inf')):
                depth.write_bytes(struct.pack('<f',value)*(640*480))
                with self.assertRaises(ValueError): load_packed(root,selection,.9)


if __name__=='__main__': unittest.main()
