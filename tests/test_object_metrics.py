"""来源：本项目原创。手算一对一匹配、错误类别、忽略区域和计数汇总用例。"""
import unittest
from drone_nav.object_metrics import parse_visdrone, evaluate_frame, aggregate, ignored_fraction, iou, size_name


def prediction(box=(0, 0, 10, 10), score=.9, group='person'):
    return dict(zip(('x1', 'y1', 'x2', 'y2'), box), score=score, group=group)


class ObjectMetricTests(unittest.TestCase):
    def labels(self, text='0,0,10,10,1,1,0,0'):
        return parse_visdrone(text, 100, 100)

    def test_duplicate_is_false_positive(self):
        r=evaluate_frame(self.labels(),[prediction(),prediction(score=.8)],100,100)['person']
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['precision']), (1,1,0,.5))
        self.assertEqual(r['duplicate_predictions'],[1])

    def test_confidence_order_and_stable_tie(self):
        r=evaluate_frame(self.labels(),[prediction(score=.5),prediction(score=.9)],100,100)['person']
        self.assertEqual(r['matches'][0]['prediction'],1)
        r=evaluate_frame(self.labels(),[prediction(),prediction()],100,100)['person']
        self.assertEqual(r['matches'][0]['prediction'],0)

    def test_wrong_class_means_false_positive_and_miss(self):
        r=evaluate_frame(self.labels(),[prediction(group='vehicle')],100,100)
        self.assertEqual(r['person']['fn'],1);self.assertEqual(r['vehicle']['fp'],1)

    def test_one_box_cannot_match_two_people(self):
        r=evaluate_frame(self.labels('0,0,10,10,1,1,0,0\n1,0,10,10,1,2,0,0'),[prediction()],100,100)['person']
        self.assertEqual((r['tp'],r['fn']),(1,1))
        self.assertEqual(r['matches'][0]['annotation'],1)

    def test_iou_threshold_and_no_plus_one(self):
        self.assertEqual(iou([0,0,10,10],[0,0,20,10]),.5)
        p=[prediction((0,0,20,10))]
        self.assertEqual(evaluate_frame(self.labels(),p,100,100,.5)['person']['tp'],1)
        self.assertEqual(evaluate_frame(self.labels(),p,100,100,.75)['person']['tp'],0)

    def test_ignored_union_not_sum_or_max(self):
        self.assertEqual(ignored_fraction([0,0,10,10],[[0,0,4,10],[3,0,7,10]]),.7)
        r=evaluate_frame(self.labels('0,0,4,10,0,0,0,0\n3,0,4,10,0,0,0,0'),[prediction()],100,100)['person']
        self.assertEqual((r['fp'],r['ignored_predictions']),(0,[0]))

    def test_exact_half_not_ignored_and_match_precedes_ignore(self):
        labels=self.labels('0,0,10,10,1,1,0,0\n0,0,5,10,0,0,0,0')
        r=evaluate_frame(labels,[prediction(),prediction(score=.8)],100,100)['person']
        self.assertEqual((r['tp'],r['fp'],r['excluded_annotations']),(1,1,[]))

    def test_targets_in_ignore_excluded_with_record(self):
        labels=self.labels('0,0,10,10,1,1,0,0\n0,0,6,10,0,0,0,0')
        r=evaluate_frame(labels,[prediction()],100,100)['person']
        self.assertEqual((r['fn'],r['excluded_annotations'],r['ignored_predictions']),(0,[1],[0]))

    def test_duplicate_cannot_hide_inside_ignore(self):
        labels=self.labels('0,0,10,10,1,1,0,0\n5,0,15,10,0,0,0,0')
        r=evaluate_frame(labels,[prediction(),prediction((2,0,12,10),.8)],100,100)['person']
        self.assertEqual(r['duplicate_predictions'],[1]);self.assertEqual(r['ignored_predictions'],[])

    def test_empty_denominators_and_pooled_counts(self):
        empty=evaluate_frame([],[],100,100)
        self.assertIsNone(empty['person']['precision']);self.assertIsNone(empty['person']['recall'])
        success=evaluate_frame(self.labels(),[prediction()],100,100)
        miss=evaluate_frame(self.labels(),[],100,100)
        r=aggregate([empty,success,miss])['person']
        self.assertEqual((r['tp'],r['fn'],r['precision'],r['recall']),(1,1,1,.5))
        self.assertEqual(r['strata']['size']['tiny']['targets'],2)

    def test_parse_original_fields_and_reject_invalid(self):
        r=self.labels('0,0,10,10,1,5,1,2,')[0]
        self.assertEqual((r['category'],r['group'],r['occlusion'],r['truncation']),(5,'vehicle',2,1))
        for text in ['0,0,0,10,1,1,0,0','-1,0,10,10,1,1,0,0','0,0,101,10,1,1,0,0','0,0,10,10,2,1,0,0','0,0,10,10,1,12,0,0','0,0,10,10,1,1,0']:
            with self.subTest(text=text),self.assertRaises(ValueError):self.labels(text)
        with self.assertRaises(ValueError):evaluate_frame(self.labels(),[prediction(score=float('nan'))],100,100)

    def test_size_boundaries(self):
        self.assertEqual([size_name([0,0,n,n]) for n in [15,16,32,96]],['tiny','small','medium','large'])


if __name__=='__main__':unittest.main()
