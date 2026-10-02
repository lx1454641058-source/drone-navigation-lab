"""来源：本项目原创。实验完整性与计时嵌套约束。"""
import copy
import unittest
from tools.vision_runtime_experiment import blocks,schedule,check_timing


class ExperimentTests(unittest.TestCase):
    def test_balanced_reversed_blocks(self):
        plan=blocks()
        self.assertEqual([(p['threads'],p['input_mode']) for p in plan[:4]],
                         [(p['threads'],p['input_mode']) for p in plan[4:]][::-1])
        calls=schedule()
        self.assertEqual(len(calls),48)
        for block in plan:
            self.assertEqual([r['index'] for r in calls if r['directory']==block['directory']],list(range(6)))

    def test_timing_rejects_omitted_work_and_invalid_measurements(self):
        row=dict(load_s=.1,model_call_s=.4,projection_s=.02,total_until_handoff_s=.52,
                 call=dict(elapsed_s=.39,prepare_s=.1,input_archive_s=.01,
                           python_stages_s=dict(png=.09,tensor=.01,input_archive=.01,worker_roundtrip=.28),
                           result=dict(inference_ms=220,worker_stages_ms=dict(inference=220,read_input=20))))
        check_timing(row)
        for key,value in [('projection_s',-.1),('total_until_handoff_s',.4),('load_s',float('nan'))]:
            changed=copy.deepcopy(row); changed[key]=value
            with self.assertRaises(ValueError): check_timing(changed)
        changed=copy.deepcopy(row)
        changed['call']['result']['worker_stages_ms']['read_input']=100
        with self.assertRaises(ValueError): check_timing(changed)


if __name__=='__main__': unittest.main()
