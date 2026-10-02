"""来源：本项目原创。研究表面图的身份、时效、边界、原子性和容量测试。"""
from copy import deepcopy
from dataclasses import asdict
import unittest

from drone_nav.frame_transform import RigidFrameTransform
from drone_nav.framed_surface_map import FramedSurfaceMap


TRANSFORM=RigidFrameTransform('source','research',((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),(0.,0.,0.),'fixture','a'*64)


def record(frame='f1',capture=1.,points=((.2,.2,.2),)):
    observations=[] if not points else [dict(detection=dict(id=0,group='person'),
        association='unverified_box_surface',object_extent_known=False,
        samples=[dict(world_m=list(p),source_world_m=list(p),pixel=[i,0],depth_z_m=2.) for i,p in enumerate(points)])]
    source=dict(flight_authorized=False,navigation_frame_alignment_available=False,
        reason='VISUAL_TARGET_HOLD' if observations else 'NO_DETECTION_REQUIRES_GEOMETRY',
        projection=dict(frame_id=frame,camera_id='c',clock_id='clock',consumer_clock_id='clock',detector_source='fixture',
            captured_at_s=capture,completed_at_s=capture+.1,consumer_at_s=capture+.2,age_s=.2,valid_until_s=capture+.5,
            world_frame='research',flight_authorized=False,free_space_evidence=[],
            status='SAMPLES_AVAILABLE' if points else 'NO_SPATIAL_SAMPLES',reasons=[],observations=observations))
    mapped=dict(converted=True,world_frame='research',source_world_frame='source',clock_id='clock',frame_id=frame,
        captured_at_s=capture,valid_until_s=capture+.5,checked_at_s=capture+.2,observation=source,transform=asdict(TRANSFORM),
        navigation_map_update_allowed=False,physical_navigation_calibrated=False,flight_authorized=False,free_space_evidence=[])
    return dict(available=True,now_s=capture+.2,age_s=.2,world_frame='research',result=mapped,
                navigation_map_update_allowed=False,physical_navigation_calibrated=False,flight_authorized=False)


class SurfaceMapTests(unittest.TestCase):
    def grid(self,cap=4096): return FramedSurfaceMap(TRANSFORM,camera_id='c',clock_id='clock',max_cells=cap)
    def put(self,grid,r,now=1.2): return grid.ingest(r,now_s=now,clock_id='clock')
    def snap(self,grid,now): return grid.snapshot(now_s=now,clock_id='clock')

    def test_negative_indices_and_boundary_protect_both_sides(self):
        grid=self.grid()
        self.assertEqual(grid.cells_for_point((-.1,.1,.1)),[(-1,0,0)])
        self.assertEqual(set(grid.cells_for_point((0,0,0))),{(x,y,z) for x in (-1,0) for y in (-1,0) for z in (-1,0)})

    def test_expiry_and_unseen_are_unknown_never_free(self):
        grid=self.grid(); self.put(grid,record())
        self.assertEqual(grid.state((0,0,0),now_s=1.5,clock_id='clock'),'current_surface_evidence')
        self.assertEqual(grid.state((0,0,0),now_s=1.5001,clock_id='clock'),'unknown_after_expiry')
        self.assertEqual(grid.state((99,99,99),now_s=1.6,clock_id='clock'),'unknown_unobserved')
        self.assertEqual(self.snap(grid,1.6)['free_cells'],0)

    def test_empty_new_frame_does_not_clear_or_refresh_old_cells(self):
        grid=self.grid(); self.put(grid,record())
        before=deepcopy(grid._cells)
        self.assertTrue(self.put(grid,record('empty',1.1,()),1.3)['accepted'])
        self.assertEqual(grid._cells,before)
        self.assertEqual(self.snap(grid,1.51)['cells'][0]['state'],'unknown_after_expiry')

    def test_duplicates_out_of_order_and_expired_rejected(self):
        grid=self.grid(); self.put(grid,record()); old=deepcopy(grid._cells)
        self.assertEqual(self.put(grid,record(),1.3)['reason'],'REPEATED_FRAME')
        self.assertEqual(self.put(grid,record('old',.9),1.3)['reason'],'OUT_OF_ORDER_CAPTURE')
        self.assertEqual(self.put(grid,record('later',1.1),1.7)['reason'],'EXPIRED_AT_RECEIPT')
        self.assertEqual(grid._cells,old)

    def test_same_world_name_different_reference_refused(self):
        r=record(); r['result']['transform']['reference_sha256']='b'*64
        grid=self.grid()
        self.assertEqual(self.put(grid,r)['reason'],'REFERENCE_TRANSFORM_MISMATCH')
        self.assertEqual(grid._cells,{})

    def test_corrupt_later_point_is_atomic(self):
        grid=self.grid(); r=record(points=((.2,.2,.2),(.3,.3,.3)))
        r['result']['observation']['projection']['observations'][0]['samples'][1]['world_m'][0]=99
        with self.assertRaises(ValueError): self.put(grid,r)
        self.assertEqual(grid._cells,{}); self.assertEqual(grid._now,-1.)

    def test_cell_budget_latches_without_partial_insert_or_clearing(self):
        grid=self.grid(1); self.put(grid,record()); old=deepcopy(grid._cells)
        self.assertEqual(self.put(grid,record('f2',1.1,((1.2,1.2,1.2),)),1.3)['reason'],'CELL_BUDGET_EXCEEDED')
        self.assertEqual(grid._cells,old)
        self.assertEqual(self.put(grid,record('f3',1.2,()),1.4)['reason'],'MAP_FAULT_LATCHED')
        self.assertEqual(grid.state((0,0,0),now_s=1.4,clock_id='clock'),'map_unavailable')
        self.assertEqual(self.snap(grid,1.4)['cells'][0]['state'],'map_unavailable')

    def test_future_conversion_and_wrong_camera_do_not_insert(self):
        grid=self.grid(); r=record()
        self.assertEqual(self.put(grid,r,1.1)['reason'],'CONVERSION_NOT_READY')
        r['result']['observation']['projection']['camera_id']='other'
        self.assertEqual(self.put(grid,r)['reason'],'CAMERA_MISMATCH')
        self.assertEqual(grid._cells,{})

    def test_snapshot_and_caller_input_do_not_alias_cells(self):
        grid=self.grid(); r=record(); self.put(grid,r)
        r['result']['observation']['projection']['observations'][0]['samples'][0]['world_m'][0]=999
        snapshot=self.snap(grid,1.2); snapshot['cells'][0]['references'][0]['world_m'][0]=999
        self.assertEqual(self.snap(grid,1.2)['cells'][0]['references'][0]['world_m'][0],.2)

    def test_clock_mismatch_and_rollback_leave_map_unchanged(self):
        grid=self.grid(); self.put(grid,record()); old=deepcopy(grid._cells)
        with self.assertRaises(ValueError): grid.ingest(record(),now_s=1.3,clock_id='wrong')
        with self.assertRaises(ValueError): self.snap(grid,1.1)
        self.assertEqual(grid._cells,old); self.assertEqual(grid._now,1.2)


if __name__=='__main__': unittest.main()
