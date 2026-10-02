"""来源：本项目原创。短周期许可与不可延期的停止回退测试。"""
from dataclasses import replace
import unittest
from drone_nav.rolling_guard import RollingConfig,IntervalObservation,RollingGuard,WatchdogExecutor,envelope


def observation(capture=0.,frame='f0'):
    return IntervalObservation(frame,'sensor','clock',capture,capture,-1.,3.,'analytical-fixture')


class RollingGuardTests(unittest.TestCase):
    def guard(self): return RollingGuard(RollingConfig(),camera_id='sensor',clock_id='clock')
    def test_prefix_includes_acceleration_reaction_and_brake(self):
        e=envelope(0.,.1,1.,.04,RollingConfig())
        self.assertAlmostEqual(e['next_speed_mps'],.12)
        self.assertAlmostEqual(e['upper_m'],.0022+.024+.0072+.34)
        self.assertAlmostEqual(e['latest_stop_s'],.38)

    def test_braking_protects_start_not_only_endpoint(self):
        e=envelope(0.,.2,-1.,0.,RollingConfig())
        self.assertAlmostEqual(e['upper_m'],.04+.02+.34)
        self.assertAlmostEqual(e['latest_stop_s'],.4)

    def test_missing_future_frame_does_not_prevent_a_supported_short_step(self):
        g=self.guard();g.receive(observation(),now_s=0.)
        ticket,r=g.choose(now_s=0.,x_m=0.,speed_mps=0.,goal_m=1.)
        self.assertIsNotNone(ticket);self.assertEqual(r['reason'],'MODEL_STEP_WITH_FALLBACK')

    def test_current_but_nearly_expired_frame_cannot_start(self):
        g=self.guard();g.receive(observation(),now_s=.46)
        ticket,r=g.choose(now_s=.46,x_m=0.,speed_mps=0.,goal_m=1.)
        self.assertIsNone(ticket);self.assertIn('EXPIRES_BEFORE_WATCHDOG_STOP',r['checks'][0]['reasons'])

    def test_stopped_past_goal_is_not_arrival(self):
        g=self.guard();g.receive(observation(),now_s=0.)
        ticket,r=g.choose(now_s=0.,x_m=1.1,speed_mps=0.,goal_m=1.)
        self.assertIsNone(ticket);self.assertEqual(r['reason'],'GOAL_PASSED_HOLD')
        self.assertEqual(g.choose(now_s=0.,x_m=1.003,speed_mps=0.,goal_m=1.)[1]['reason'],'GOAL_STOPPED')

    def test_identity_coverage_duplicate_and_order_faults_preserve_evidence(self):
        for bad in (replace(observation(.1,'f1'),camera_id='wrong'),replace(observation(.1,'f1'),clock_id='wrong'),
                    replace(observation(.1,'f1'),coverage_model='point_samples'),observation(.1,'f0'),observation(0.,'old')):
            with self.subTest(bad=bad):
                g=self.guard(); original=observation();g.receive(original,now_s=0.)
                self.assertFalse(g.receive(bad,now_s=.1)['accepted']);self.assertEqual(g.latest,original)
                self.assertIsNone(g.choose(now_s=.1,x_m=0.,speed_mps=0.,goal_m=1.)[0])

    def test_watchdog_stops_when_controller_sends_nothing(self):
        g=self.guard();g.receive(observation(),now_s=0.);e=WatchdogExecutor(RollingConfig(),camera_id='sensor',clock_id='clock')
        for _ in range(5):
            t,_=g.choose(now_s=e.now_s,x_m=e.x_m,speed_mps=e.speed_mps,goal_m=1.);e.step(t)
        bound=e.committed['bound'];deadline=e.committed['ticket']['observation']['captured_at_s']+.5
        for _ in range(30): e.step(None)
        self.assertEqual(e.speed_mps,0.);self.assertLessEqual(e.x_m+.34,bound['upper_m']+1e-9)
        self.assertLessEqual(e.fallback_at_s+.2+.1,deadline+1e-9)

    def test_repeated_none_and_late_tickets_cannot_reset_fallback(self):
        g=self.guard();g.receive(observation(),now_s=0.);e=WatchdogExecutor(RollingConfig(),camera_id='sensor',clock_id='clock')
        t,_=g.choose(now_s=0.,x_m=0.,speed_mps=0.,goal_m=1.);e.step(t);e.step(None)
        started=e.fallback_at_s
        for _ in range(20): e.step(t)
        self.assertEqual(e.fallback_at_s,started);self.assertEqual(e.speed_mps,0.)

    def test_wrong_executor_state_ticket_rejected_atomically(self):
        g=self.guard();g.receive(observation(),now_s=0.);t,_=g.choose(now_s=0.,x_m=0.,speed_mps=0.,goal_m=1.)
        e=WatchdogExecutor(RollingConfig(),camera_id='sensor',clock_id='clock')
        with self.assertRaises(ValueError): e.step(replace(t,x_m=.1))
        self.assertEqual((e.now_s,e.x_m,e.speed_mps),(0.,0.,0.))

    def test_executor_rechecks_clock_and_nonfinite_ticket(self):
        g=self.guard();g.receive(observation(),now_s=0.);t,_=g.choose(now_s=0.,x_m=0.,speed_mps=0.,goal_m=1.)
        for bad in (replace(t,at_s=float('nan')),replace(t,observation=replace(t.observation,clock_id='wrong'))):
            e=WatchdogExecutor(RollingConfig(),camera_id='sensor',clock_id='clock')
            with self.assertRaises(ValueError): e.step(bad)
            self.assertEqual(e.now_s,0.)

    def test_expired_future_and_rollback_input(self):
        g=self.guard();self.assertFalse(g.receive(observation(),now_s=.51)['accepted'])
        g=self.guard();self.assertFalse(g.receive(observation(.1,'future'),now_s=0.)['accepted'])
        g=self.guard();g.receive(observation(),now_s=.1)
        with self.assertRaises(ValueError): g.choose(now_s=0.,x_m=0.,speed_mps=0.,goal_m=1.)


if __name__=='__main__': unittest.main()
