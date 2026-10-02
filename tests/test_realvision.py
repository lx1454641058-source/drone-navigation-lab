"""来源：本项目原创。检查评估分母、类别转换、坏输入与采样算法，避免虚高成绩。"""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from drone_nav.realvision import (PROJECT, UAVID_MAP, ade_mapping, confusion, metrics,
                                  run_suite, aggregate)
from drone_nav.verify_realvision import read_maps
from tools.setup_vision import contained, put


class EvaluationTests(unittest.TestCase):
    def test_known_confusion_not_average_image_accuracy(self):
        m = metrics(confusion(bytes([1, 1, 2, 6]), bytes([1, 2, 2, 0])))
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['person_recall'], 0)
        self.assertEqual(m['nonroad_as_road_fraction'], 1 / 3)
        self.assertEqual(m['per_class'][1]['iou'], .5)
        self.assertEqual(m['mean_iou'], .25)

    def test_absent_class_is_none_not_success(self):
        m = metrics(confusion(b'\x02\x02', b'\x02\x02'))
        self.assertIsNone(m['person_recall'])
        self.assertIsNone(m['nonroad_as_road_fraction'])
        self.assertEqual(m['iou_class_count'], 1)

    def test_false_positive_only_class_counts_in_mean(self):
        m = metrics(confusion(b'\0\0', b'\0\6'))
        self.assertEqual(m['per_class'][6]['iou'], 0)
        self.assertIsNone(m['per_class'][6]['recall'])
        self.assertEqual(m['mean_iou'], .25)

    def test_invalid_maps_rejected(self):
        for a, b in [(b'', b''), (b'\0', b''), (b'\7', b'\0'), ([False], [0])]:
            with self.assertRaises(ValueError):
                confusion(a, b)

    def test_invalid_counts_rejected(self):
        for matrix in [[], [[0] * 7] * 7, [[-1] * 7] * 7, [[True] * 7] * 7]:
            with self.assertRaises(ValueError):
                metrics(matrix)

    def test_vehicles_merge_without_motion_claim(self):
        self.assertEqual(UAVID_MAP[3], UAVID_MAP[7])
        self.assertEqual(UAVID_MAP[0], 0)

    def test_aggregate_weights_pixels(self):
        cases = []
        for split in ('development', 'validation'):
            for t, p in [(b'\1', b'\1'), (b'\1' * 9, b'\2' * 9)]:
                m = metrics(confusion(t, p))
                cases.append(dict(split=split, metrics=dict(segformer=m, color=m)))
        result = aggregate(cases)['validation']['segformer']
        self.assertEqual(result['accuracy'], .1)
        self.assertEqual(result['pixel_count'], 10)

    def test_unknown_taxonomy_rejected(self):
        with self.assertRaises(ValueError):
            ade_mapping({'id2label': {'0': 'unknown'}})

    def test_locked_sequences_do_not_overlap(self):
        lock = json.loads((PROJECT / 'tools/vision_assets.json').read_text(encoding='utf-8'))
        groups = {s: {c['sequence'] for c in lock['cases'] if c['split'] == s}
                  for s in ('development', 'validation')}
        self.assertFalse(groups['development'] & groups['validation'])
        self.assertEqual(len(groups['validation']), 7)
        self.assertEqual(len({c['image'] for c in lock['cases']}), 9)

    def test_existing_output_refused_before_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run_suite(Path(directory))

    def test_download_refuses_overwrite_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            put(root / 'sample', b'original')
            put(root / 'sample', b'original')
            with self.assertRaises(ValueError):
                put(root / 'sample', b'changed')
            for name in ('../outside', '/outside', 'C:/outside', 'a\\b'):
                with self.assertRaises(ValueError):
                    contained(root, name)
            self.assertEqual((root / 'sample').read_bytes(), b'original')

    def test_changed_mapped_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('ade', 'uavid', 'truth', 'segformer', 'color', 'color-original'):
                (root / (name + '.u8')).write_bytes(b'\0')
            with self.assertRaisesRegex(ValueError, 'mapping'):
                read_maps(root, 1, bytes(range(150)))

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for image math tests')
    def test_node_image_math(self):
        result = subprocess.run(['node', str(PROJECT / 'tests/realvision_math.cjs')],
                                check=True, capture_output=True, text=True)
        self.assertIn('image math passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
