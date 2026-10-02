"""来源：本项目原创。移动观测、物理回执及故障停止的边界测试。"""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from drone_nav.native_physics import DEFAULT_DLL
from drone_nav.occupancy import VoxelMap
from drone_nav.pinhole import Intrinsics,PerspectiveFrame,Pose
from drone_nav.physical_navigation import navigate_physical
from drone_nav.physical_vehicle import FlightBudget,PhysicalVehicle,physical_guard,world_xml
from drone_nav.raycast import Box,Surface,World
from drone_nav.verify_coupled import RecordedPhysicalVehicle,normalize
from drone_nav.vision_experiment import train_model


class MovingMapTests(unittest.TestCase):
    def frames(self):
        k=Intrinsics(2,1,1000,1000,.5,0)
        return [PerspectiveFrame(k,Pose.look_at((1.5,y,3.5),(8,y,3.5)),((100,100,100),)*2,(4,4),0)
                for y in (2.5,5.5)]

    def test_each_ray_uses_own_origin(self):
        grid=VoxelMap(10,8);grid.integrate_moving(self.frames(),0)
        for y in (2,5):
            self.assertEqual(grid.state((3,y,3),0),'free')
            self.assertEqual(grid.state((5,y,3),0),'occupied')
        self.assertEqual(grid.state((2,3,3),0),'unknown')

    def test_old_stationary_protocol_still_rejects_moving_batch(self):
        frames=self.frames()
        with self.assertRaises(ValueError):VoxelMap(10,8).integrate(frames,0,frames[0].pose.position)

    def test_moving_batch_invalid_tail_is_atomic(self):
        for bad in (replace(self.frames()[1],tick=1),replace(self.frames()[1],depth_z_m=(31,31))):
            grid=VoxelMap(10,8)
            with self.assertRaises(ValueError):grid.integrate_moving([self.frames()[0],bad],0)
            self.assertEqual((grid.free_seen,grid.occupied,grid.last_tick),({},set(),-1))

    def test_guard_requires_whole_reserved_volume_and_freshness(self):
        grid=VoxelMap(20,16);budget=FlightBudget()
        grid.free_seen={(x,y,3):0 for x in range(20) for y in range(16)}
        self.assertTrue(physical_guard(grid,(3,8),(4,8),0,.9,{0:0},budget)['allowed'])
        self.assertEqual(physical_guard(grid,(3,8),(4,8),0,4,{0:0},budget)['reason'],'STALE_MAP_HOLD')
        del grid.free_seen[(4,9,3)]
        self.assertEqual(physical_guard(grid,(3,8),(4,8),0,.9,{0:0},budget)['reason'],'BRAKE_MARGIN_HOLD')

    def test_invalid_flight_budget(self):
        for values in ({'free_ttl_s':9.4},{'body_radius_m':.45},{'stable_s':9},{'position_tolerance_m':.2},{'max_move_s':float('nan')}):
            with self.assertRaises(ValueError):FlightBudget(**values)

    def test_world_adapter_preserves_boxes_and_rejects_slopes(self):
        ground=Surface('g',0,(-40,60,-40,56));box=Box('b',3,(8,6,0),(10,10,6))
        root=ET.fromstring(world_xml(World((ground,box))))
        geom=root.find("worldbody/geom[@name='obstacle_0']")
        self.assertEqual([float(v) for v in geom.get('pos').split()],[9,8,3])
        self.assertEqual([float(v) for v in geom.get('size').split()],[1,2,3])
        with self.assertRaises(ValueError):world_xml(World((replace(ground,slope_x=.1),)))


@unittest.skipUnless(DEFAULT_DLL.exists(),'optional verified native physics runtime unavailable')
class CoupledPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world=World((Surface('g',0,(-40,60,-40,56)),));cls.model=train_model()
        vehicle=PhysicalVehicle(cls.world,(3,8))
        try:
            cls.result=navigate_physical(vehicle,cls.model,(3,8),(4,8),previews=False)
            cls.actions=vehicle.actions;cls.commands=vehicle.commands
            cls.frames=json.loads(json.dumps(vehicle.frames))
        finally:vehicle.close()

    def test_arrival_acknowledges_position_and_speed_without_snapping(self):
        self.assertEqual(self.result['terminal_state'],'READY_TO_LAND')
        receipt=self.actions[0]
        self.assertTrue(receipt['completed']);self.assertTrue(receipt['stop_confirmed'])
        self.assertGreater(receipt['position_error_m'],.0001)
        self.assertLessEqual(receipt['position_error_m'],.03)
        self.assertLessEqual(receipt['speed_mps'],.03)
        self.assertGreaterEqual(receipt['stable_duration_s'],.3)
        self.assertGreater(receipt['duration_s'],2)

    def test_camera_tracks_continuous_physics_during_scan(self):
        for item in self.frames:
            self.assertEqual(item['frame']['pose']['position'],item['capture_state']['position'])
        self.assertNotEqual(self.frames[8]['frame']['pose']['position'],self.frames[9]['frame']['pose']['position'])
        self.assertAlmostEqual(self.frames[9]['capture_state']['time_s']-self.frames[8]['capture_state']['time_s'],.1)

    def test_saved_images_reproduce_feedback_without_rendering(self):
        vehicle=RecordedPhysicalVehicle(self.world,(3,8),self.frames)
        try:
            with patch('drone_nav.physical_vehicle.render',side_effect=AssertionError('must not render')):
                replay=navigate_physical(vehicle,self.model,(3,8),(4,8),previews=False)
            self.assertEqual(normalize(replay),normalize(self.result))
            self.assertEqual(vehicle.commands,self.commands)
            self.assertEqual(vehicle.frame_index,len(self.frames))
        finally:vehicle.close()

    def test_ignored_instruction_cannot_advance_confirmed_cell(self):
        vehicle=PhysicalVehicle(self.world,(3,8),ignore_move=0)
        try:
            r=navigate_physical(vehicle,self.model,(3,8),(4,8),previews=False)
            self.assertEqual(r['terminal_state'],'CONTROLLER_TIMEOUT')
            self.assertEqual(r['trace'][-1]['position'],(3,8))
            self.assertIsNone(r['landing']);self.assertEqual(len(vehicle.actions),1)
            self.assertFalse(vehicle.actions[0]['completed'])
            self.assertTrue(vehicle.actions[0]['stop_confirmed'])
        finally:vehicle.close()

    def test_depth_loss_does_not_issue_another_move(self):
        vehicle=PhysicalVehicle(self.world,(3,8),dropout_tick=1)
        try:
            r=navigate_physical(vehicle,self.model,(3,8),(5,8),previews=False)
            self.assertEqual(r['terminal_state'],'SENSOR_HOLD')
            self.assertEqual(len(vehicle.actions),1);self.assertIsNone(r['landing'])
        finally:vehicle.close()

    def test_invalid_scan_stops_before_any_move(self):
        vehicle=PhysicalVehicle(self.world,(3,8))
        try:
            with patch.object(vehicle,'scan',return_value=[]):
                r=navigate_physical(vehicle,self.model,(3,8),(4,8),previews=False)
            self.assertEqual(r['terminal_state'],'SENSOR_PROTOCOL_HOLD')
            self.assertEqual(vehicle.commands,[]);self.assertIsNone(r['landing'])
        finally:vehicle.close()

    def test_landing_camera_error_never_accepts(self):
        vehicle=PhysicalVehicle(self.world,(3,8));capture=vehicle.capture
        def bad(k,tick,aim=None):
            frame=capture(k,tick,aim)
            return replace(frame,tick=tick+1) if aim is None else frame
        try:
            with patch.object(vehicle,'capture',side_effect=bad):
                r=navigate_physical(vehicle,self.model,(3,8),(3,8),previews=False)
            self.assertEqual(r['terminal_state'],'LANDING_SENSOR_HOLD');self.assertIsNone(r['landing'])
        finally:vehicle.close()


if __name__=='__main__':unittest.main()
