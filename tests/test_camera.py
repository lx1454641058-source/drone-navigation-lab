"""来源：本项目原创。用解析几何和异常输入验证相机，不依赖渲染图的目测。"""

import math
import unittest
from dataclasses import replace

from drone_nav.camera_experiment import error_summary
from drone_nav.pinhole import Intrinsics, PerspectiveFrame, Pose, observed_grid, reconstruct
from drone_nav.raycast import Box, Surface, World, demo_world, first_hit, render


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.k = Intrinsics(5, 5, 2, 2, 2, 2)
        self.pose = Pose.look_at((0, 0, 10), (0, 0, 0))

    def test_known_world_coordinates_project_with_correct_axis_signs(self):
        # 垂直朝下时右侧为世界 +x，图像下侧为世界 -y。
        self.assertEqual(self.pose.project((5, 5, 0), self.k), (3, 1, 10))
        self.assertEqual(self.pose.unproject(3, 1, 10, self.k), (5, 5, 0))

    def test_oblique_center_ray_hits_expected_point(self):
        pose = Pose.look_at((0, 0, 10), (10, 0, 0))
        actual = pose.unproject(2, 2, math.sqrt(200), self.k)
        for a, b in zip(actual, (10, 0, 0)):
            self.assertAlmostEqual(a, b, places=10)

    def test_projected_size_changes_with_distance(self):
        near = self.pose.project((1, 0, 5), self.k)[0]-self.k.cx
        far = self.pose.project((1, 0, 0), self.k)[0]-self.k.cx
        self.assertAlmostEqual(near, 2*far)

    def test_behind_camera_is_not_projectable(self):
        self.assertIsNone(self.pose.project((0, 0, 11), self.k))
        self.assertIsNone(self.pose.project(self.pose.position, self.k))

    def test_slant_range_cannot_be_used_as_z_depth(self):
        correct = self.pose.unproject(4, 2, 10, self.k)
        wrong = self.pose.unproject(4, 2, math.sqrt(200), self.k)
        self.assertEqual(correct, (10, 0, 0))
        self.assertGreater(math.dist(wrong, correct), 5)

    def test_roundtrip_varied_rotations_and_pixels(self):
        for position, target in [((3, 4, 9), (1, 2, 0)), ((2, 0, 5), (3, 6, 1)),
                                 ((0, 0, 0), (0, 0, 1))]:
            pose = Pose.look_at(position, target)
            for u, v, z in [(0, 4, 3), (2, 2, 12), (3.1, .7, 8)]:
                observed = pose.project(pose.unproject(u, v, z, self.k), self.k)
                for a, b in zip(observed, (u, v, z)):
                    self.assertAlmostEqual(a, b, places=10)

    def test_bad_intrinsics_poses_and_depth_are_rejected(self):
        for change in ({"fx": 0}, {"fy": math.nan}, {"width": True}, {"cx": 100}, {"height": 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.k, **change)
        for change in ({"right": (2, 0, 0)}, {"down": (0, 1, 0)}, {"position": (math.nan, 0, 1)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.pose, **change)
        for z in (0, -1, math.nan, True):
            with self.assertRaises(ValueError):
                self.pose.unproject(0, 0, z, self.k)
        with self.assertRaises(ValueError):
            Pose.look_at((0, 0, 0), (0, 0, 0))


class RayTests(unittest.TestCase):
    def test_plane_depth_has_closed_form(self):
        plane = Surface("ramp", 0, (-20, 20, -20, 20), z0=1, slope_x=.2)
        # z=10-t, x=t，因此 10-t=1+.2t，得到 t=7.5。
        self.assertAlmostEqual(plane.intersect((0, 0, 10), (1, 0, -1)), 7.5)
        self.assertIsNone(plane.intersect((0, 0, 10), (0, 1, 0)))
        self.assertIsNone(plane.intersect((50, 0, 10), (0, 0, -1)))

    def test_box_parallel_inside_and_behind_rays(self):
        box = Box("box", 3, (1, 1, 0), (3, 3, 4))
        self.assertEqual(box.intersect((2, 2, 10), (0, 0, -1)), 6)
        self.assertEqual(box.intersect((2, 2, 2), (1, 0, 0)), 1)
        self.assertIsNone(box.intersect((0, 2, 10), (0, 0, -1)))
        self.assertIsNone(box.intersect((2, 2, 10), (0, 0, 1)))

    def test_nearest_surface_wins_independent_of_scene_order(self):
        ground = Surface("ground", 0, (-10, 10, -10, 10))
        box = Box("roof", 3, (-1, -1, 0), (1, 1, 5))
        for surfaces in ((ground, box), (box, ground)):
            depth, surface = first_hit(World(surfaces), (0, 0, 10), (0, 0, -1))
            self.assertEqual(depth, 5)
            self.assertEqual(surface.name, "roof")

    def test_building_hides_ground_and_side_view_reveals_it(self):
        world = demo_world()
        probe = (11, 8, 0)
        for position, name in (((5, 8, 8), "building"), ((12, 4, 8), "ground")):
            direction = tuple(b-a for a, b in zip(position, probe))
            depth, surface = first_hit(world, position, direction)
            self.assertEqual(surface.name, name)
            self.assertLess(depth, 1) if name == "building" else self.assertAlmostEqual(depth, 1)

    def test_flat_ground_z_depth_is_constant_for_downward_camera(self):
        k = Intrinsics(5, 5, 3, 3, 2, 2)
        pose = Pose.look_at((0, 0, 10), (0, 0, 0))
        frame, _ = render(World((Surface("ground", 0, (-30, 30, -30, 30)),)), pose, k)
        self.assertEqual(frame.depth_z_m, (10,)*25)
        self.assertTrue(all(abs(point[2]) < 1e-10 for _, point in reconstruct(frame)))

    def test_physical_max_range_is_not_z_depth_limit(self):
        k = Intrinsics(3, 3, 1, 1, 1, 1)
        pose = Pose.look_at((0, 0, 10), (0, 0, 0))
        frame, _ = render(World((Surface("ground", 0, (-30, 30, -30, 30)),)),
                          pose, k, max_range_m=11)
        self.assertEqual(frame.depth_z_m[4], 10)
        self.assertIsNone(frame.depth_z_m[0])  # 角点距离为 sqrt(300) 米。

    def test_render_reproducibility_with_noise_and_dropout(self):
        args = (demo_world(), Pose.look_at((12, 4, 8), (11, 8, 0)), Intrinsics(8, 6, 6, 6, 3.5, 2.5))
        self.assertEqual(render(*args, seed=909, dropout=.2, noise_std_m=.05),
                         render(*args, seed=909, dropout=.2, noise_std_m=.05))

    def test_invalid_geometry_and_render_settings(self):
        with self.assertRaises(ValueError):
            Box("bad", 0, (1, 1, 1), (0, 0, 0))
        with self.assertRaises(ValueError):
            Surface("bad", 0, (1, 0, 0, 1))
        for kwargs in ({"dropout": -1}, {"dropout": 2}, {"light": math.nan}, {"noise_std_m": -1}):
            with self.assertRaises(ValueError):
                render(demo_world(), Pose.look_at((0, 0, 8), (0, 0, 0)), Intrinsics(), **kwargs)


class PointCloudTests(unittest.TestCase):
    def setUp(self):
        self.k = Intrinsics(3, 3, 2, 2, 1, 1)
        self.pose = Pose.look_at((5, 5, 8), (5, 5, 0))

    def test_missing_depth_produces_no_observations(self):
        frame = PerspectiveFrame(self.k, self.pose, ((120, 120, 120),)*9, (None,)*9, 0)
        points = reconstruct(frame)
        self.assertEqual(points, [])
        grid = observed_grid(points, [0]*9, width=40, height=32)
        self.assertEqual(grid["unknown_cells"], 1280)
        self.assertEqual(grid["cells"], [])

    def test_sparse_endpoints_do_not_fill_space_between_them(self):
        grid = observed_grid([(0, (1.1, 1.1, 0)), (1, (3.1, 1.1, 2))], [0, 3],
                             width=10, height=10, resolution_m=1)
        self.assertEqual(grid["unknown_cells"], 98)
        self.assertEqual([(c["x"], c["y"]) for c in grid["cells"]], [(1, 1), (3, 1)])
        self.assertEqual([c["kind"] for c in grid["cells"]], ["surface", "raised"])

    def test_unknown_label_remains_uncertain(self):
        grid = observed_grid([(0, (1, 1, 0))], [-1], width=10, height=10)
        self.assertEqual(grid["cells"][0]["kind"], "uncertain")

    def test_invalid_frame_is_rejected_before_reconstruction(self):
        frame = PerspectiveFrame(self.k, self.pose, ((1, 2, 3),)*9, (8,)*9, 0)
        for change in ({"tick": False}, {"rgb": ((1, 2, 3),)}, {"depth_z_m": (True,)*9},
                       {"depth_z_m": (math.nan,)*9}, {"rgb": ((-1, 0, 0),)*9}):
            with self.assertRaises(ValueError):
                reconstruct(replace(frame, **change))

    def test_empty_error_sample_is_not_zero_error(self):
        self.assertEqual(error_summary([]), {"count": 0, "mean_m": None, "p95_m": None, "max_m": None})


if __name__ == "__main__":
    unittest.main()
