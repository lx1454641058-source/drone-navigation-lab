"""来源：本项目原创。实验对照不能绕过身份、时效或几何检查。"""
import unittest
from dataclasses import replace
from drone_nav.detection_bridge import DetectionPacket
from drone_nav.textured_veto import finish_visual_veto


class TexturedVetoTests(unittest.TestCase):
    def packet(self):
        return DetectionPacket('frame','simulation-rgbd',160,120,(),'test-fixture',1.,1.2,'physics-seconds')

    def test_positive_target_stops_before_geometry(self):
        result=finish_visual_veto(dict(reason='VISUAL_TARGET_HOLD'),self.packet(),now_s=1.3)
        self.assertEqual(result['reason'],'VISUAL_TARGET_HOLD')
        self.assertFalse(result['permit_geometry_check']); self.assertFalse(result['flight_authorized'])

    def test_ablation_only_removes_target_veto_and_still_requires_geometry(self):
        result=finish_visual_veto(dict(reason='VISUAL_TARGET_HOLD'),self.packet(),now_s=1.3,veto_enabled=False)
        self.assertEqual(result['reason'],'DEVELOPMENT_TARGET_VETO_DISABLED')
        self.assertTrue(result['permit_geometry_check']); self.assertFalse(result['flight_authorized'])

    def test_ablation_never_removes_context_failure(self):
        result=finish_visual_veto(dict(reason='VISUAL_CONTEXT_HOLD'),self.packet(),now_s=1.3,veto_enabled=False)
        self.assertFalse(result['permit_geometry_check'])

    def test_processing_crossing_expiry_rejects_both_arms(self):
        for enabled in (True,False):
            result=finish_visual_veto(dict(reason='NO_DETECTION_REQUIRES_GEOMETRY'),self.packet(),now_s=1.50001,veto_enabled=enabled)
            self.assertEqual(result['reason'],'VISUAL_EXPIRED_AT_COMMIT')
            self.assertFalse(result['permit_geometry_check'])

    def test_invalid_final_time_or_non_boolean_arm_rejects(self):
        for changes in (dict(now_s=1.1),dict(now_s=float('nan')),dict(now_s=1.3,veto_enabled=0)):
            with self.assertRaises(ValueError):finish_visual_veto(dict(reason='VISUAL_TARGET_HOLD'),self.packet(),**changes)

    def test_missing_or_foreign_clock_rejects_even_in_ablation(self):
        for packet in (replace(self.packet(),captured_at_s=None,completed_at_s=None),
                       replace(self.packet(),clock_id='foreign-clock')):
            with self.assertRaises(ValueError):
                finish_visual_veto(dict(reason='VISUAL_TARGET_HOLD'),packet,now_s=1.3,veto_enabled=False)


if __name__=='__main__': unittest.main()
