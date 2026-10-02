"""来源：本项目原创。选图隔离与拥挤目标去重损失，不调用模型。"""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from tools.whole_detection_study import select_reserved, evaluate
from drone_nav.object_metrics import parse_visdrone


class WholeDetectionTests(unittest.TestCase):
    def inventory(self):
        names=[f'VisDrone2019-DET-val/images/{i//4:07d}_{i%4:05d}.jpg' for i in range(548)]
        representatives=names[::4]
        ordered=sorted(representatives,key=lambda n:hashlib.sha256(('drone-nav-object-dev-v1:'+Path(n).name).encode()).hexdigest())
        return names,ordered

    def test_selection_is_deterministic_and_disjoint(self):
        names,ordered=self.inventory()
        result=select_reserved(list(reversed(names)),ordered[:12])
        self.assertEqual(result,ordered[12:24])
        self.assertTrue(set(Path(n).stem.split('_')[0] for n in result).isdisjoint(Path(n).stem.split('_')[0] for n in ordered[:12]))

    def test_invalid_inventory_or_development_list_rejected(self):
        names,ordered=self.inventory()
        for bad,old in [(names[:-1],ordered[:12]),(names[:-1]+[names[0]],ordered[:12]),(names,ordered[1:13])]:
            with self.assertRaises(ValueError):select_reserved(bad,old)

    def test_too_few_prefixes_rejected(self):
        names=[f'VisDrone2019-DET-val/images/{i//46:07d}_{i%46:05d}.jpg' for i in range(548)]
        representatives=names[::46]
        old=sorted(representatives,key=lambda n:hashlib.sha256(('drone-nav-object-dev-v1:'+Path(n).name).encode()).hexdigest())
        with self.assertRaises(ValueError):select_reserved(names,old)

    def test_crowded_real_targets_lost_by_actual_dedup(self):
        labels=parse_visdrone('0,0,10,10,1,1,0,0\n2,0,10,10,1,2,0,0',100,100)
        boxes=[dict(x1=x,y1=0,x2=x+10,y2=10,score=score,group='person') for x,score in [(0,.9),(2,.8)]]
        js="const {dedup}=require('./tools/whole_detection_worker.cjs'); console.log(JSON.stringify(dedup(JSON.parse(require('fs').readFileSync(0,'utf8')))));"
        proc=subprocess.run(['node','-e',js],input=json.dumps(boxes),capture_output=True,text=True,check=True,timeout=30,cwd=Path(__file__).resolve().parent.parent)
        clean=json.loads(proc.stdout)
        report=evaluate([dict(key='crowd',width=100,height=100,annotations=labels)],
                        [dict(key='crowd',arm=arm,boxes=b) for arm,b in [('tinyformer-1',boxes),('tinyformer-nms',clean['boxes'])]])
        self.assertEqual(report['summary']['tinyformer-1']['0.50']['person']['tp'],2)
        self.assertEqual(report['summary']['tinyformer-nms']['0.50']['person']['tp'],1)
        self.assertEqual(report['transitions'][0]['thresholds']['0.50']['person'],dict(recovered=[],lost=[2],retained=[1]))

    def test_javascript_boundaries(self):
        p=subprocess.run(['node',str(Path(__file__).with_name('whole_detection_worker.cjs'))],capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)


if __name__=='__main__':unittest.main()
