"""来源：本项目原创。框覆盖的像素中心、去重计数及已有成果保护。"""
import subprocess
import tempfile
from pathlib import Path
import unittest
from tools.probe_detection import coverage,run,PROJECT
from tools.verify_detection import reference_coverage


class DetectionProbeTests(unittest.TestCase):
    def test_javascript_geometry_and_bgr(self):
        r=subprocess.run(['node',str(PROJECT/'tests/detection_math.cjs')],check=True,capture_output=True,text=True)
        self.assertIn('detection math passed',r.stdout)

    def test_union_counts_overlap_once_and_keeps_pixel_centers(self):
        b=dict(group='person',x1=.5,y1=0,x2=2.5,y2=1)
        r=coverage(bytes([6,0,6,6]),[b,b],4,1,4,1)['person']
        self.assertEqual(r['box_union_pixels'],2)
        self.assertEqual(r['covered_label_pixels'],1)
        self.assertEqual(r['label_coverage'],1/3)

    def test_absent_class_and_no_boxes(self):
        r=coverage(bytes([0]),[],1,1,1,1)['person']
        self.assertIsNone(r['label_coverage']);self.assertIsNone(r['target_fraction_in_boxes'])

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):run(Path(d))

    def test_nonfinite_box_rejected(self):
        with self.assertRaises(ValueError):
            coverage(bytes([6]),[dict(group='person',x1=0,y1=0,x2=float('nan'),y2=1)],1,1,1,1)

    def test_independent_oracle_checks_scaled_edges_overlap_and_groups(self):
        # 源图放大两倍：评价中心为 1、3、5、7；右边界 5 不包含中心 5。
        b=dict(group='person',x1=-2,y1=0,x2=5,y2=2)
        boxes=[b,b,dict(group='vehicle',x1=5,y1=0,x2=9,y2=2)]
        r=reference_coverage(bytes([6,0,6,3]),boxes,4,1,8,2)
        self.assertEqual(r['person']['box_union_pixels'],2)
        self.assertEqual(r['person']['covered_label_pixels'],1)
        self.assertEqual(r['person']['label_coverage'],.5)
        self.assertEqual(r['vehicle']['box_union_pixels'],2)
        self.assertEqual(r['vehicle']['covered_label_pixels'],1)

    def test_present_label_without_boxes_is_miss_not_absent_sample(self):
        r=reference_coverage(bytes([6]),[],1,1,1,1)['person']
        self.assertEqual(r['label_coverage'],0)
        self.assertIsNone(r['target_fraction_in_boxes'])
        self.assertEqual(r,coverage(bytes([6]),[],1,1,1,1)['person'])


if __name__=='__main__':unittest.main()
