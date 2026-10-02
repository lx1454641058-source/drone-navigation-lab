"""来源：本项目原创。位置身份合并与第三批固定选图的边界验证。"""
import hashlib
from pathlib import Path
import unittest

from drone_nav.query_dedup import merge_queries
from drone_nav.object_metrics import parse_visdrone,evaluate_frame
from drone_nav.detection_failures import changes
from tools.query_detection_study import select_third


def box(query=0,score=.9,group='person',tile=0,x=0):
    return dict(x1=x,y1=0,x2=x+10,y2=10,query=query,score=score,group=group,tile=tile)


class QueryTests(unittest.TestCase):
    def test_highest_score_and_stable_tie(self):
        r=merge_queries([box(score=.6),box(score=.9),box(score=.9)])
        self.assertEqual(r['kept_predictions'],[1])
        self.assertEqual([x['prediction'] for x in r['removed']],[2,0])
        self.assertTrue(all(x['kept_prediction']==1 for x in r['removed']))
        self.assertEqual(merge_queries([]),dict(boxes=[],kept_predictions=[],removed=[]))

    def test_distinct_identity_and_group_retained(self):
        original=[box(),box(query=1),box(tile=1),box(group='vehicle')]
        self.assertEqual(merge_queries(original)['kept_predictions'],[0,1,2,3])
        self.assertEqual(len(original),4)

    def test_inconsistent_geometry_and_missing_identity_rejected(self):
        with self.assertRaises(ValueError):merge_queries([box(),box(x=1)])
        for field,value in [('query',None),('query',300),('tile',-1),('tile',True),('score',float('nan')),('score',1.1),('group','other'),('x2',0)]:
            b=box();b[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):merge_queries([b])

    def test_neighboring_queries_keep_both_targets(self):
        labels=parse_visdrone('0,0,10,10,1,1,0,0\n2,0,10,10,1,2,0,0',100,100)
        boxes=[box(),box(query=1,x=2,score=.8)]
        m=evaluate_frame(labels,merge_queries(boxes)['boxes'],100,100)['person']
        self.assertEqual((m['tp'],m['fp'],m['fn']),(2,0,0))

    def test_identical_query_can_lose_a_match_in_crowded_labels(self):
        labels=parse_visdrone('0,0,10,10,1,1,0,0\n2,0,10,10,1,2,0,0',100,100)
        boxes=[box(),box(score=.8)]
        a=evaluate_frame(labels,boxes,100,100)['person'];b=evaluate_frame(labels,merge_queries(boxes)['boxes'],100,100)['person']
        self.assertEqual(changes(a,b),dict(recovered=[],lost=[2],retained=[1]))

    def test_third_selection_order_and_old_batches(self):
        names=[f'VisDrone2019-DET-val/images/{i//4:07d}_{i%4:05d}.jpg' for i in range(548)]
        order=sorted(names[::4],key=lambda n:hashlib.sha256(('drone-nav-object-dev-v1:'+Path(n).name).encode()).hexdigest())
        self.assertEqual(select_third(names[::-1],order[:12],order[12:24]),order[24:36])
        with self.assertRaises(ValueError):select_third(names,order[:12],order[13:25])
        with self.assertRaises(ValueError):select_third(names[:-1],order[:12],order[12:24])

    def test_not_enough_third_prefixes_rejected(self):
        names=[f'VisDrone2019-DET-val/images/{i//20:07d}_{i%20:05d}.jpg' for i in range(548)]
        order=sorted(names[::20],key=lambda n:hashlib.sha256(('drone-nav-object-dev-v1:'+Path(n).name).encode()).hexdigest())
        with self.assertRaises(ValueError):select_third(names,order[:12],order[12:24])


if __name__=='__main__':unittest.main()
