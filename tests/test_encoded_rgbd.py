"""来源：本项目原创。PNG/帧绑定必须拒绝错图与可变输入，保持原张量。"""
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from drone_nav.pinhole import Intrinsics,Pose
from drone_nav.packed_rgbd import PackedFrame
from drone_nav.encoded_rgbd import EncodedFrame,EncodedVisionSession,load_encoded
from tools.tinyformer_probe import prepare


def png(image,fmt='PNG'):
    stream=BytesIO(); image.save(stream,format=fmt); return stream.getvalue()


class EncodedTests(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(3,2),(31,127,255))
        self.frame=PackedFrame(Intrinsics(3,2,2,2,1,.5),
            Pose((0,0,0),(1,0,0),(0,1,0),(0,0,1)),self.image.tobytes(),(2.,)*6)

    def test_immutable_binding_and_exact_archive_bytes(self):
        encoded=EncodedFrame(self.frame,png(self.image))
        self.assertEqual(encoded.png_bytes,png(self.image))
        with self.assertRaises(FrozenInstanceError): encoded.png_bytes=b''
        with self.assertRaises(FrozenInstanceError): encoded.frame=self.frame

    def test_wrong_pixels_dimensions_mode_format_and_mutable_input_rejected(self):
        bad=(png(Image.new('RGB',(3,2),(32,127,255))),png(Image.new('RGB',(2,3))),
             png(self.image.convert('RGBA')),png(self.image,'BMP'),b'not-png',
             bytearray(png(self.image)),png(self.image)[:30])
        for payload in bad:
            with self.subTest(payload_type=type(payload)):
                with self.assertRaises(ValueError): EncodedFrame(self.frame,payload)

    def test_file_change_between_frame_load_and_archive_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'rgb.png').write_bytes(png(Image.new('RGB',(3,2),(1,2,3))))
            with patch('drone_nav.encoded_rgbd.load_packed',return_value=SimpleNamespace(frame=self.frame)):
                with self.assertRaises(ValueError): load_encoded(root,dict(rgb=(1.,'rgb.png')),0)

    def test_bad_session_input_fails_before_creating_archive(self):
        session=EncodedVisionSession.__new__(EncodedVisionSession)
        session.replay=False; session.input_mode='raw'
        with self.assertRaises(ValueError): session.detect_encoded(self.frame)
        encoded=EncodedFrame(self.frame,png(self.image))
        session.input_mode='gzip'
        with self.assertRaises(ValueError): session.detect_encoded(encoded)
        session.input_mode='raw'; session.replay=True
        with self.assertRaises(ValueError): session.detect_encoded(encoded)

    def test_archived_tensor_matches_reference_and_png_matches_bound_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            session=EncodedVisionSession.__new__(EncodedVisionSession)
            session.replay=False; session.input_mode='raw'; session.threads=16
            session.output=Path(d); session.records=[]
            session._request=lambda request: dict(boxes=[])
            encoded=EncodedFrame(self.frame,png(self.image))
            record=session.detect_encoded(encoded)
            directory=Path(d)/record['directory']
            self.assertEqual((directory/'input.png').read_bytes(),encoded.png_bytes)
            self.assertEqual((directory/'input.f32').read_bytes(),prepare(self.image,record['window']))


if __name__=='__main__': unittest.main()
