"""来源：本项目原创。候选筛查的像素口径与成果保护边界。"""
from pathlib import Path
import tempfile
import unittest
from tools.probe_cityscapes import binary_metrics, run


class CandidateProbeTests(unittest.TestCase):
    def test_binary_counts_and_false_positive(self):
        m=binary_metrics(bytes([6,6,0,0]),bytes([11,0,11,0]),6,(11,))
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,1,1))
        self.assertEqual(m['recall'],.5)
        self.assertEqual(m['precision'],.5)
        self.assertAlmostEqual(m['iou'],1/3)

    def test_absent_ground_truth_is_not_perfect_recall(self):
        m=binary_metrics(b'\0',b'\x0b',6,(11,))
        self.assertIsNone(m['recall'])
        self.assertEqual(m['iou'],0)

    def test_invalid_shapes(self):
        for a,b in [(b'',b''),(b'\0',b'')]:
            with self.assertRaises(ValueError):
                binary_metrics(a,b,6,(11,))

    def test_existing_archive_rejected_before_any_download(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                run(Path(d),Path('missing-reference'))


if __name__=='__main__':
    unittest.main()
