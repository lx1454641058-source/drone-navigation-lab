"""来源：本项目原创。实际推力提交处的时效、状态、目标和期限反例。"""
from dataclasses import replace
import unittest
from drone_nav.actuator_vision import VisualMotionLease,submission_check


class SubmissionTests(unittest.TestCase):
    def lease(self):return VisualMotionLease('frame',(4.5,8.5,3.5),1.,1.3,20.)
    def check(self,**changes):
        args=dict(lease=self.lease(),target=(4.5,8.5,3.5),now_s=1.3,dt=.002,
            clock_id='physics-seconds',command_state={'time_s':1.3},current_state={'time_s':1.3},move_started_at_s=1.3)
        args.update(changes)
        if 'now_s' in changes:
            for field in ('command_state','current_state'):
                if field not in changes:args[field]={'time_s':changes['now_s']}
        return submission_check(**args)
    def test_current_step_is_eligible_but_no_flight_claim(self):
        r=self.check();self.assertTrue(r['allowed']);self.assertFalse(r['flight_authorized'])
    def test_last_full_step_allowed_and_crossing_step_rejected(self):
        self.assertTrue(self.check(now_s=1.498)['allowed'])
        self.assertEqual(self.check(now_s=1.499)['reason'],'VISUAL_EXPIRED_AT_ACTUATOR')
    def test_exact_expiry_is_too_late_to_start_next_step(self):
        self.assertFalse(self.check(now_s=1.5)['allowed'])
    def test_foreign_clock_rejects_before_age_comparison(self):
        self.assertEqual(self.check(clock_id='other')['reason'],'ACTUATOR_CLOCK_MISMATCH')
    def test_target_change_does_not_reuse_lease(self):
        self.assertEqual(self.check(target=(3.5,9.5,3.5))['reason'],'ACTUATOR_TARGET_MISMATCH')
    def test_same_position_later_control_state_rejects(self):
        self.assertEqual(self.check(now_s=1.302,command_state={'time_s':1.3})['reason'],'CONTROL_STATE_CHANGED_BEFORE_SUBMISSION')
    def test_geometry_must_cover_full_move_and_stop(self):
        self.assertEqual(self.check(lease=replace(self.lease(),geometry_valid_until_s=9.7))['reason'],'GEOMETRY_EXPIRES_BEFORE_STOP')
    def test_missing_or_invalid_times_reject(self):
        self.assertFalse(self.check(lease=None)['allowed'])
        for args in (dict(now_s=float('nan')),dict(dt=0.),dict(now_s=1.2),dict(move_started_at_s=2.)):
            self.assertFalse(self.check(**args)['allowed'])
    def test_expired_lease_cannot_be_constructed(self):
        for args in (dict(available_at_s=1.6),dict(clock_id='wrong'),dict(target=[4.5,8.5,3.5])):
            with self.assertRaises(ValueError):replace(self.lease(),**args)

    def test_state_and_reported_time_must_agree(self):
        self.assertEqual(self.check(current_state={'time_s':9.})['reason'],'ACTUATOR_STATE_INVALID')

    def test_new_observation_cannot_extend_original_motion_duration(self):
        lease=replace(self.lease(),captured_at_s=10.,available_at_s=10.3)
        self.assertEqual(self.check(lease=lease,now_s=10.3)['reason'],'MOTION_DURATION_EXPIRED')


if __name__=='__main__':unittest.main()
