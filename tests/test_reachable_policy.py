"""来源：本项目原创。验证可停止有向图搜索、复杂布局及同条件对照。"""

from copy import deepcopy
import unittest

from drone_nav.localization import MislocalizedCamera,audit_localization
from drone_nav.localization_experiment import compare_policies,localization_cases
from drone_nav.motion import MotionConfig,movement_guard,profile
from drone_nav.occupancy import VoxelMap
from drone_nav.reachable_policy import ReachableFrontierPolicy
from drone_nav.timed_navigation import navigate
from drone_nav.verify_exploration import RecordedCamera


class ReachableGraphTests(unittest.TestCase):
    def route(self,edges,start=(1,1),goal=(3,1)):
        grid = VoxelMap(6,5)
        policy = ReachableFrontierPolicy()
        path,state,_ = policy.route(grid,start,goal,0,
            edge_check=lambda a,b:{'allowed':(a,b) in edges,'reason':'BRAKE_MARGIN_HOLD'},
            can_stay=lambda c:True)
        return path,state,policy.diagnostics

    def test_blocked_direct_edge_uses_allowed_detour(self):
        expected = [(1,1),(1,2),(2,2),(3,2),(3,1)]
        edges = set(zip(expected,expected[1:]))
        path,state,info = self.route(edges)
        self.assertEqual(path,expected)
        self.assertEqual(state,'GOAL_PATH')
        self.assertGreater(info['rejected_space'],0)

    def test_edges_are_directional(self):
        path,state,_ = self.route({((2,1),(1,1)),((3,1),(2,1))})
        self.assertEqual(path,[])
        self.assertEqual(state,'NO_FEASIBLE_FRONTIER')

    def test_shortest_allowed_path(self):
        straight = [(1,1),(2,1),(3,1)]
        longer = [(1,1),(1,2),(2,2),(3,2),(3,1)]
        path,state,_ = self.route(set(zip(straight,straight[1:]))|set(zip(longer,longer[1:])))
        self.assertEqual(path,straight)

    def test_all_stale_edges_leave_no_route(self):
        policy = ReachableFrontierPolicy()
        path,state,_ = policy.route(VoxelMap(6,5),(1,1),(3,1),0,
            edge_check=lambda a,b:{'allowed':False,'reason':'STALE_MAP_HOLD'},can_stay=lambda c:True)
        self.assertEqual(path,[])
        self.assertEqual(state,'NO_FEASIBLE_FRONTIER')
        self.assertEqual(policy.diagnostics['rejected_age'],4)

    def test_unknown_stationary_footprint_prevents_departure(self):
        def unexpected(a,b): self.fail('must not inspect edges without a supported starting footprint')
        path,state,_ = ReachableFrontierPolicy().route(VoxelMap(6,5),(1,1),(3,1),0,
            edge_check=unexpected,can_stay=lambda c:False)
        self.assertEqual((path,state),([],'NO_OBSERVED_START'))

    def test_scan_without_gain_does_not_cancel_reachable_new_viewpoint(self):
        grid = VoxelMap(10,10)
        grid.free_seen = {(x,y,3):0 for x in range(1,6) for y in range(1,6)}
        policy = ReachableFrontierPolicy()
        for _ in range(4): policy.update(grid,(2,2))
        self.assertEqual(policy.no_gain_scans,3)
        c = MotionConfig()
        path,state,_ = policy.route(grid,(2,2),(8,8),0,can_stay=lambda c:True,
            edge_check=lambda a,b:movement_guard(grid,a,b,0,.9,{0:0},profile(1,c),c))
        self.assertEqual(state,'FRONTIER_PATH')
        self.assertGreater(len(path),1)
        self.assertNotIn(path[-1],policy.scanned)

    def test_search_work_is_finite_on_cycles(self):
        grid = VoxelMap(6,5)
        policy = ReachableFrontierPolicy()
        path,_,_ = policy.route(grid,(1,1),(3,1),0,
            edge_check=lambda a,b:{'allowed':True,'reason':'APPROVED'},can_stay=lambda c:True)
        self.assertEqual(policy.diagnostics['reachable_cells'],30)
        self.assertLessEqual(policy.diagnostics['checked_edges'],4*30)
        self.assertEqual(len(path),3)


class ReachableNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for case in (localization_cases()[6],localization_cases()[8]):
            camera = MislocalizedCamera(case['world'],case['error'],record=True)
            result = navigate(camera,20,16,case['start'],case['goal'],localization=case['budget'],
                              policy_mode='reachable',previews=False)
            cls.results[case['key']] = (case,result,camera.frames)

    def test_staggered_ideal_now_reaches_goal(self):
        case,r,_ = self.results['staggered_ideal']
        self.assertEqual(r['terminal_state'],'ARRIVED_WAYPOINT')
        self.assertEqual(r['trace'][-1]['position'],case['goal'])
        audit = audit_localization(case['world'],r,case['error'],case['budget'])
        self.assertTrue(audit['inside_goal_tolerance'])
        self.assertEqual(audit['collision_segments'],0)
        self.assertEqual(audit['boundary_violations'],0)

    def test_bounded_building_can_move_but_still_cannot_confirm_arrival(self):
        case,r,_ = self.results['building_bounded']
        self.assertGreater(r['distance_m'],1)
        self.assertEqual(r['terminal_state'],'UNCERTAIN_ARRIVAL')
        self.assertEqual(r['localization_budget']['axis_bound_m'],.45)
        audit = audit_localization(case['world'],r,case['error'],case['budget'])
        self.assertEqual(audit['collision_segments'],0)
        self.assertFalse(audit['false_arrival'])

    def test_approved_steps_have_bounded_search_and_valid_timing(self):
        for _,r,_ in self.results.values():
            self.assertTrue(all(m['guard']['allowed'] for m in r['moves']))
            self.assertTrue(all(m['end_s']<=m['guard']['latest_stop_s'] for m in r['moves']))
            self.assertTrue(all(t['planning_diagnostics']['checked_edges']<=4*20*16 for t in r['trace']))

    def test_new_policy_replays_from_observations(self):
        case,saved,frames = self.results['building_bounded']
        replay = navigate(RecordedCamera(frames),20,16,case['start'],case['goal'],localization=case['budget'],
                          policy_mode='reachable',previews=False)
        self.assertEqual(replay,saved)


class PairedConditionsTests(unittest.TestCase):
    def fixture(self):
        return {'key':'test','title':'test','world':{},'error_model':{},'localization_budget':{'axis_bound_m':.45},
                'config':{},'intrinsics':{},'start':(1,1),'goal':(5,1),'max_ticks':70,'seed':1201,
                'terminal_state':'NO_FEASIBLE_FRONTIER','distance_m':3,
                'audit':{'inside_goal_tolerance':False,'false_arrival':False}}

    def test_changed_error_bound_cannot_be_counted_as_strategy_improvement(self):
        old = self.fixture()
        new = deepcopy(old)
        new['localization_budget']['axis_bound_m'] = 0
        with self.assertRaisesRegex(ValueError,'localization_budget'): compare_policies([new],[old])

    def test_missing_or_duplicate_cases_rejected(self):
        r = self.fixture()
        with self.assertRaises(ValueError): compare_policies([r],[])
        with self.assertRaises(ValueError): compare_policies([r,r],[r,r])

    def test_outcomes_recorded_without_equating_stops_and_arrivals(self):
        old = self.fixture()
        new = deepcopy(old)
        new['terminal_state'] = 'UNCERTAIN_ARRIVAL'
        new['distance_m'] = 13
        result = compare_policies([new],[old])[0]
        self.assertFalse(result['after_actual_arrival'])
        self.assertEqual(result['after_terminal'],'UNCERTAIN_ARRIVAL')


if __name__=='__main__': unittest.main()
