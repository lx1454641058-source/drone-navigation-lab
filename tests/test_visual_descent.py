"""来源：本项目原创。降落授权、证据时效、故障中断和接地确认回归。"""

from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from drone_nav.descent_experiment import descent_cases,run_case
from drone_nav.descent_vehicle import DescentVehicle
from drone_nav.native_physics import DEFAULT_DLL,QuadrotorPhysics
from drone_nav.physical_vehicle import world_xml
from drone_nav.verify_descent import RecordedDescentVehicle,normalize_descent,replay_raw
from drone_nav.physical_navigation import navigate_physical
from drone_nav.visual_descent import DescentConfig,contact_candidate,descend_after_navigation,envelope_reason
from drone_nav.vision_experiment import train_model


class DescentContractTests(unittest.TestCase):
    def test_config_rejects_invalid_duration_geometry_and_limits(self):
        for values in ({'rate_mps':float('nan')},{'corridor_radius_m':.4},{'observation_period_s':.08},
                       {'processing_s':.003},{'max_speed_mps':.1},{'disarmed_settle_s':.1},{'contact_gap_m':.05}):
            with self.assertRaises(ValueError):DescentConfig(**values)

    def state(self):
        return dict(position=[4.5,8.5,.06],velocity=[0,0,0],quaternion=[1,0,0,0],angular_velocity=[0,0,0])

    def test_contact_requires_more_than_height(self):
        c=DescentConfig();s=self.state()
        self.assertTrue(contact_candidate(s,0,(4.5,8.5),c))
        for gap in (None,float('nan'),.1,-.02):self.assertFalse(contact_candidate(s,gap,(4.5,8.5),c))
        for key,value in (('velocity',[0,0,-.1]),('position',[4.5,8.5,.5]),('angular_velocity',[1,0,0]),('position',[4.7,8.5,.06])):
            bad=deepcopy(s);bad[key]=value
            self.assertFalse(contact_candidate(bad,0,(4.5,8.5),c))

    def test_envelope_rejects_speed_and_lateral_drift(self):
        s=self.state();s['velocity'][0]=.5
        self.assertEqual(envelope_reason(s,(4.5,8.5),DescentConfig()),'EXCESSIVE_SPEED')
        s=self.state();s['position'][0]+=.2
        self.assertEqual(envelope_reason(s,(4.5,8.5),DescentConfig()),'LATERAL_DRIFT')


@unittest.skipUnless(DEFAULT_DLL.exists(),'optional verified native runtime unavailable')
class VisualDescentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=train_model();cls.case=descent_cases()[0]
        cls.result,raw=run_case(cls.case,cls.model,previews=False)
        cls.raw=json.loads(json.dumps(raw))

    def test_navigation_then_visual_descent_and_disarmed_touchdown(self):
        r=self.result;d=r['descent']
        self.assertEqual(r['navigation']['terminal_state'],'READY_TO_LAND')
        self.assertEqual(d['status'],'LANDED');self.assertTrue(d['stop_confirmed'])
        self.assertGreater(d['disarmed_at_s'],d['started_at_s'])
        self.assertAlmostEqual(d['ended_at_s']-d['disarmed_at_s'],1)
        self.assertAlmostEqual(d['final_state']['position'][2],.06,delta=.002)
        self.assertTrue(all(m==[0.0]*4 for m in self.raw['commands'][-500:]))
        self.assertGreater(len(d['events']),20)

    def test_local_view_does_not_refresh_whole_corridor_evidence(self):
        events=self.result['descent']['events'];local=[e for e in events if not e['full_corridor']]
        self.assertTrue(local);self.assertLess(local[-1]['visible_radius_m'],.1)
        self.assertTrue(all(e['evidence']['accepted'] for e in events))
        trace=[t for t in self.result['descent']['trace'] if t['phase']=='TERMINAL_LOCAL_VIEW']
        self.assertTrue(trace)
        self.assertGreater(trace[-1]['certificate_age_s'],trace[0]['certificate_age_s'])
        self.assertLessEqual(max(t['certificate_age_s'] for t in trace),8)

    def test_saved_images_and_probes_replay_without_rendering(self):
        v=RecordedDescentVehicle(self.case['world'],(3,8),self.raw)
        try:
            with patch('drone_nav.physical_vehicle.render',side_effect=AssertionError('must use saved frames')):
                nav=navigate_physical(v,self.model,(3,8),(4,8),previews=False)
                d=descend_after_navigation(v,self.model,nav,previews=False)
            self.assertEqual(normalize_descent(d),normalize_descent(self.result['descent']))
            self.assertEqual(v.commands,self.raw['commands'])
            self.assertEqual(v.frame_index,len(self.raw['frames']))
            self.assertEqual(v.contact_index,len(self.raw['contact_readings']))
        finally:v.close()

    def test_raw_thrust_and_probe_order_reproduce_physics(self):
        with QuadrotorPhysics(model_xml=world_xml(self.case['world'])) as q:
            history=replay_raw(q,self.raw,(3,8))
        self.assertEqual(history,self.result['history'])

    def test_rejected_and_stale_authorization_issue_no_motors(self):
        v=DescentVehicle(self.case['world'],(3,8))
        try:
            for terminal in ('LANDING_REJECTED','READY_TO_LAND'):
                nav=deepcopy(self.result['navigation']);nav['terminal_state']=terminal
                d=descend_after_navigation(v,self.model,nav,previews=False)
                self.assertEqual(d['status'],'NOT_AUTHORIZED')
                self.assertEqual(v.commands,[])
        finally:v.close()

    def run_failure(self,key):
        case=next(c for c in descent_cases() if c['key']==key)
        return run_case(case,self.model,previews=False)

    def test_water_never_starts_descent(self):
        r,raw=self.run_failure('water')
        self.assertEqual(r['descent']['status'],'NOT_AUTHORIZED')
        self.assertEqual(r['descent']['events'],[]);self.assertEqual(raw['contact_readings'],[])

    def test_same_color_bump_rejected_by_tighter_corridor_check(self):
        r,raw=self.run_failure('bump')
        self.assertEqual(r['navigation']['terminal_state'],'READY_TO_LAND')
        self.assertEqual(r['descent']['status'],'ABORTED_HOLD')
        self.assertEqual(r['descent']['reason'],'VISION_REJECTED')
        self.assertIn('UNEVEN_SURFACE',r['descent']['events'][0]['evidence']['reason_codes'])
        self.assertGreater(r['actual_final']['position'][2],3.4)

    def test_depth_and_semantic_failure_abort_without_disarming(self):
        for key in ('depth','semantic'):
            r,raw=self.run_failure(key);d=r['descent']
            self.assertEqual(d['status'],'ABORTED_HOLD',key)
            self.assertEqual(d['reason'],'VISION_REJECTED',key)
            self.assertIsNone(d['disarmed_at_s']);self.assertTrue(d['stop_confirmed'])
            self.assertGreater(sum(raw['commands'][-1]),10)
            self.assertGreater(r['actual_final']['position'][2],2)

    def test_missing_contact_evidence_cannot_land(self):
        r,raw=self.run_failure('contact');d=r['descent']
        self.assertEqual(d['status'],'ABORTED_HOLD')
        self.assertEqual(d['reason'],'CONTACT_SENSOR_INVALID')
        self.assertIsNone(d['disarmed_at_s'])
        self.assertIsNone(raw['contact_readings'][-1]['value'])

    def test_expired_outer_corridor_aborts_even_if_local_image_is_clear(self):
        r,raw=self.run_failure('expired');d=r['descent']
        self.assertEqual(d['status'],'ABORTED_HOLD')
        self.assertEqual(d['reason'],'CORRIDOR_EVIDENCE_EXPIRED')
        self.assertTrue(d['events'][-1]['evidence']['accepted'])
        self.assertFalse(d['events'][-1]['full_corridor'])
        self.assertIsNone(d['disarmed_at_s'])
        active=[t for t in d['trace'] if t['phase'] in ('DESCEND','TERMINAL_LOCAL_VIEW') and t['certificate_age_s'] is not None]
        self.assertLessEqual(max(t['certificate_age_s'] for t in active),1+1e-8)

    def test_old_evidence_must_cover_image_processing_delay(self):
        r,_=run_case(dict(self.case,config=DescentConfig(certificate_ttl_s=.55)),self.model,previews=False)
        d=r['descent']
        self.assertEqual(d['reason'],'CORRIDOR_EVIDENCE_EXPIRED')
        self.assertEqual(d['status'],'ABORTED_HOLD')
        self.assertEqual(len(d['events']),1)
        abort=next(t for t in d['trace'] if t['phase']=='ABORT_HOLD')
        self.assertAlmostEqual(abort['state']['time_s']-d['started_at_s'],.5)

    def test_invalid_descent_frame_never_reaches_motor_cut(self):
        v=DescentVehicle(self.case['world'],(3,8))
        try:
            nav=navigate_physical(v,self.model,(3,8),(4,8),previews=False);capture=v.capture
            def bad(k,tick,aim=None):return replace(capture(k,tick,aim),tick=tick+1)
            with patch.object(v,'capture',side_effect=bad):d=descend_after_navigation(v,self.model,nav,previews=False)
            self.assertEqual(d['reason'],'VISION_PROTOCOL_ERROR');self.assertIsNone(d['disarmed_at_s'])
            self.assertEqual(d['status'],'ABORTED_HOLD')
        finally:v.close()

    def test_post_disarm_probe_failure_does_not_claim_landed(self):
        case=dict(self.case,fault='contact',fault_after_s=27.5)
        r,raw=run_case(case,self.model,previews=False)
        self.assertEqual(r['descent']['status'],'TOUCHDOWN_UNCONFIRMED')
        self.assertIsNotNone(r['descent']['disarmed_at_s'])
        self.assertFalse(r['descent']['stop_confirmed'])

    def test_timeout_and_insufficient_stop_time_remain_failures(self):
        for config,expected in ((DescentConfig(max_duration_s=.5),'ABORTED_HOLD'),
                                (DescentConfig(max_duration_s=.5,abort_hold_s=.1),'ABORT_STOP_UNCONFIRMED')):
            r,_=run_case(dict(self.case,config=config),self.model,previews=False)
            self.assertEqual(r['descent']['status'],expected)
            self.assertEqual(r['descent']['reason'],'DESCENT_TIMEOUT')
            self.assertIsNone(r['descent']['disarmed_at_s'])


if __name__=='__main__':unittest.main()
