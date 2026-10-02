"""来源：本项目原创。分块数学、独立对照、指标聚合及参数只读解析边界。"""

import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from drone_nav.realvision import PROJECT, metrics, confusion
from drone_nav.scale_experiment import ARMS, summarize, run_suite
from tools.inspect_vision_weights import fields, varint, transpose, f32, inspect


class ScaleTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js required')
    def test_tile_math_and_independent_whole_reference(self):
        result = subprocess.run(['node', str(PROJECT / 'tests/segmentation_tiles_math.cjs')],
                                capture_output=True, text=True, check=True)
        self.assertIn('tile math passed', result.stdout)

    def test_aggregate_keeps_split_arm_and_pixel_weights(self):
        cases = []
        for split in ('development', 'validation'):
            for n, correct in [(1, True), (9, False)]:
                entries = {}
                for arm in ARMS:
                    pred = b'\1' * n if correct or arm == 'grid4' else b'\2' * n
                    entries[arm] = dict(metrics=metrics(confusion(b'\1' * n, pred)), worker=dict(core_ms=n))
                cases.append(dict(split=split, arms=entries))
        result = summarize(cases)
        self.assertEqual(result['validation']['whole']['accuracy'], .1)
        self.assertEqual(result['validation']['grid4']['accuracy'], 1)
        self.assertEqual(result['development']['whole']['median_core_ms'], 5)

    def test_existing_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run_suite(Path(directory))

    def test_proto_scalar_and_bytes(self):
        self.assertEqual(list(fields(b'\x08\x96\x01\x12\x03abc')), [(1, 150), (2, b'abc')])

    def test_truncated_proto_rejected(self):
        for data in (b'\x80', b'\x12\x05ab', b'\x0d\x00', b'\0\0', b'\x0b'):
            with self.assertRaises(ValueError):
                list(fields(data))

    def test_varint_bounds(self):
        self.assertEqual(varint(b'\xff' * 9 + b'\x01', 0), (2**64 - 1, 10))
        with self.assertRaises(ValueError):
            varint(b'\xff' * 9 + b'\x02', 0)

    def test_float_transpose_uses_values_not_reversed_bytes(self):
        raw = struct.pack('<6f', 1, 2, 3, 4, 5, 6)
        self.assertEqual(struct.unpack('<6f', transpose(raw, 2, 3)), (1, 4, 2, 5, 3, 6))
        with self.assertRaises(ValueError):
            transpose(raw, 3, 3)

    def test_float32_rounding(self):
        self.assertEqual(f32(16777217), 16777216)

    def test_other_models_cannot_pass_parameter_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            a.write_bytes(b'not the fixed model')
            b.write_bytes(b'not the fixed reference')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                inspect(a, b)


if __name__ == '__main__':
    unittest.main()
