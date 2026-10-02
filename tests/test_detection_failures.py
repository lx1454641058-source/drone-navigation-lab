"""来源：本项目原创。诊断优先级和找回/丢失目标不能互相掩盖。"""
import subprocess
from pathlib import Path
import unittest
from drone_nav.detection_failures import diagnose,crosses_seam,changes


class FailureTests(unittest.TestCase):
    target=dict(id=1,box=[0,0,10,10],group='person',occlusion=0,truncation=0)
    box=dict(x1=0,y1=0,x2=10,y2=10,group='person',score=.9,tile=0,audit_id='0:0',owned=True)

    def trace(self,box,final=False):
        return dict(candidates=[box],variants={'baseline':[box] if final else []},
                    suppressions={'baseline':[dict(removed=box['audit_id'],kept='0:1',iou=.6)]})

    def test_javascript_trace(self):
        r=subprocess.run(['node',str(Path(__file__).with_name('trace_detection.cjs'))],capture_output=True,text=True,check=True)
        self.assertIn('trace detection checks passed',r.stdout)

    def test_final_matching_priority(self):
        self.assertEqual(diagnose(self.target,self.trace(self.box,True))['reason'],'matching_competition')

    def test_nms_and_core_stage_separate(self):
        r=diagnose(self.target,self.trace(self.box))
        self.assertEqual(r['reason'],'nms_support_removed');self.assertEqual(r['evidence']['suppression']['kept'],'0:1')
        self.assertEqual(diagnose(self.target,self.trace(dict(self.box,owned=False)))['reason'],'core_support_removed')

    def test_low_score_not_counted_as_high_score(self):
        self.assertEqual(diagnose(self.target,self.trace(dict(self.box,score=.2)))['reason'],'low_score_support')

    def test_localization_and_wrong_group(self):
        self.assertEqual(diagnose(self.target,self.trace(dict(self.box,x2=30)))['reason'],'localization_support')
        self.assertEqual(diagnose(self.target,self.trace(dict(self.box,group='vehicle')))['reason'],'other_group_overlap')

    def test_no_qualifying_support(self):
        self.assertEqual(diagnose(self.target,self.trace(dict(self.box,x1=50,x2=60)))['reason'],'no_qualifying_support')

    def test_seam_strict_border_and_odd_dimensions(self):
        self.assertTrue(crosses_seam([1,0,3,2],9,9,4))
        self.assertFalse(crosses_seam([0,0,2,2],9,9,4))
        self.assertFalse(crosses_seam([0,0,9,9],9,9,1))

    def test_recovered_and_lost_are_both_retained(self):
        r=changes({'matches':[{'annotation':1},{'annotation':2}]},{'matches':[{'annotation':2},{'annotation':3}]})
        self.assertEqual(r,dict(recovered=[3],lost=[1],retained=[2]))


if __name__=='__main__':unittest.main()
