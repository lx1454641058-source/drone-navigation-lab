"""来源：本项目原创。验证像素到决策的接口、几何估计、模型与失败行为。"""

import json
import math
import struct
import unittest
import zlib
from dataclasses import replace

from drone_nav.classifier import ColorModel
from drone_nav.image_sensor import ImageSensor
from drone_nav.imaging import CLASSES, RGBDFrame, png_bytes
from drone_nav.mission import LandingSite, Observation, Scenario, run_mission
from drone_nav.perception import analyze_frame, landing_evidence
from drone_nav.rendering import OrthographicCamera, training_image
from drone_nav.vision_experiment import TRAIN_SEEDS, TEST_SEEDS, SHIFT_SEEDS, train_model, vision_scenarios


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train_model()

    def test_training_and_evaluation_seeds_do_not_overlap(self):
        self.assertFalse(set(TRAIN_SEEDS) & set(TEST_SEEDS))
        self.assertFalse(set(TRAIN_SEEDS) & set(SHIFT_SEEDS))
        self.assertFalse(set(TEST_SEEDS) & set(SHIFT_SEEDS))

    def test_model_serialization_preserves_predictions_and_hash(self):
        recovered = ColorModel.from_dict(json.loads(json.dumps(self.model.to_dict())))
        pixels, _ = training_image(210)
        self.assertEqual(self.model.digest, recovered.digest)
        self.assertEqual(self.model.predict_image(pixels), recovered.predict_image(pixels))

    def test_model_learns_labels_not_fixed_palette_ids(self):
        pixels, labels = training_image(150)
        shifted = [(label+1) % len(CLASSES) for label in labels]
        model = ColorModel.fit([(pixels, shifted)])
        predicted = model.predict_image(pixels)
        accuracy = sum(a == b for a, b in zip(predicted, shifted))/len(shifted)
        self.assertGreater(accuracy, 0.98)

    def test_unknown_and_black_pixels_are_rejected(self):
        for pixel in ((0, 0, 0), (208, 35, 197)):
            self.assertEqual(self.model.predict(pixel), -1)

    def test_corrupt_model_is_rejected(self):
        for corruption in ("variance", "nan", "classes", "count", "threshold"):
            data = json.loads(json.dumps(self.model.to_dict()))
            if corruption == "variance": data["variances"][0][0] = 0
            elif corruption == "nan": data["means"][0][0] = math.nan
            elif corruption == "classes": data["classes"][0] = "invented"
            elif corruption == "count": data["counts"][0] = True
            else: data["reject_distance"] = -1
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                ColorModel.from_dict(data)

    def test_invalid_training_data(self):
        for images in ([], [([(1, 2, 3)], [])], [([(1, 2, 3)], [5])]):
            with self.assertRaises(ValueError):
                ColorModel.fit(images)


class ImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train_model()

    def frame(self, slope=0.0, occupied=False, surface="paved"):
        s = Scenario("frame", "frame", "frame", sites=[LandingSite("A", (15, 10),
                     slope_deg=slope, occupied=occupied, surface=surface)])
        return OrthographicCamera(s).capture((15, 10), 0)

    def analyze(self, frame):
        return analyze_frame(frame, self.model, (30, 20), 3.0)

    def test_plane_slope_comes_from_depth(self):
        for angle in (0, 4, 15):
            frame = self.frame(slope=angle)
            evidence = landing_evidence(frame, self.analyze(frame), (15, 10))
            self.assertAlmostEqual(evidence["slope_deg"], angle, delta=0.1)
            self.assertEqual(evidence["accepted"], angle < 5)

    def test_rgb_change_alone_changes_landing_decision(self):
        paved = self.frame()
        water = self.frame(surface="water")
        # 保留完全相同的深度、位置和标定，只改变颜色像素。
        changed = replace(paved, rgb=water.rgb)
        self.assertTrue(landing_evidence(paved, self.analyze(paved), (15, 10))["accepted"])
        evidence = landing_evidence(changed, self.analyze(changed), (15, 10))
        self.assertFalse(evidence["accepted"])
        self.assertEqual(evidence["surface"], "water")

    def test_geometry_alone_accepts_flat_water_but_semantics_rejects(self):
        frame = self.frame(surface="water")
        analysis = self.analyze(frame)
        self.assertTrue(landing_evidence(frame, analysis, (15, 10), semantic=False)["accepted"])
        self.assertFalse(landing_evidence(frame, analysis, (15, 10))["accepted"])

    def test_occupant_is_rejected(self):
        frame = self.frame(occupied=True)
        evidence = landing_evidence(frame, self.analyze(frame), (15, 10))
        self.assertFalse(evidence["accepted"])
        self.assertTrue(evidence["occupied"])

    def test_unknown_surface_is_rejected(self):
        frame = self.frame(surface="unknown")
        evidence = landing_evidence(frame, self.analyze(frame), (15, 10))
        self.assertFalse(evidence["accepted"])

    def test_missing_depth_inside_landing_footprint_rejects(self):
        frame = self.frame()
        depth = list(frame.depth_m)
        depth[(frame.height//2)*frame.width+frame.width//2] = math.nan
        frame = replace(frame, depth_m=tuple(depth))
        self.assertFalse(landing_evidence(frame, self.analyze(frame), (15, 10))["accepted"])

    def test_incomplete_footprint_rejects(self):
        frame = self.frame()
        self.assertFalse(landing_evidence(frame, self.analyze(frame), (9, 10))["accepted"])

    def test_depth_obstacle_mapping_from_pixels(self):
        frame = self.frame()
        depth = list(frame.depth_m)
        for y in range(6, 9):
            for x in range(6, 9):
                depth[y*frame.width+x] = 1.0
        changed = replace(frame, depth_m=tuple(depth))
        expected = (frame.origin_cell[0]+2, frame.origin_cell[1]+2)
        self.assertIn(expected, self.analyze(changed).obstacles)
        self.assertNotIn(expected, self.analyze(frame).obstacles)

    def test_no_depth_becomes_blocked_not_free(self):
        frame = self.frame()
        changed = replace(frame, depth_m=(math.nan,)*len(frame.depth_m))
        analysis = self.analyze(changed)
        self.assertEqual(analysis.valid_fraction, 0)
        self.assertIn((15, 10), analysis.obstacles)

    def test_corrupt_frame_shapes_and_pixels(self):
        frame = self.frame()
        for changed in (replace(frame, rgb=frame.rgb[:-1]), replace(frame, width=0),
                        replace(frame, pixels_per_cell=0), replace(frame, camera_altitude_m=math.nan),
                        replace(frame, rgb=((-1, 0, 0),)*len(frame.rgb))):
            with self.assertRaises(ValueError):
                self.analyze(changed)

    def test_png_roundtrip_structure_and_pixels(self):
        pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (12, 34, 56)]
        blob = png_bytes(2, 2, pixels)
        self.assertEqual(blob[:8], b"\x89PNG\r\n\x1a\n")
        offset = 8
        image_data = b""
        while offset < len(blob):
            length = struct.unpack("!I", blob[offset:offset+4])[0]
            kind = blob[offset+4:offset+8]
            body = blob[offset+8:offset+8+length]
            checksum = struct.unpack("!I", blob[offset+8+length:offset+12+length])[0]
            self.assertEqual(zlib.crc32(kind+body), checksum)
            if kind == b"IDAT": image_data += body
            offset += length+12
        expected = b"\0\xff\0\0\0\xff\0\0\0\0\xff\x0c\x22\x38"
        self.assertEqual(zlib.decompress(image_data), expected)


class VisionMissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train_model()

    def test_camera_stale_frame_stops_without_motion(self):
        scenario = vision_scenarios()[0]
        sensor = ImageSensor(scenario, self.model, preview=False)
        frame = sensor.camera.capture(scenario.start, 2)
        class StaleCamera:
            def capture(self, position, tick): return frame
        sensor.camera = StaleCamera()
        result = run_mission(scenario, sensor)
        self.assertEqual(result["metrics"]["terminal_state"], "SENSOR_HOLD")
        self.assertEqual(result["metrics"]["grid_moves"], 0)

    def test_full_image_navigation_and_evidence(self):
        scenario = vision_scenarios()[4]
        result = run_mission(scenario, ImageSensor(scenario, self.model, preview=False))
        self.assertEqual(result["metrics"]["terminal_state"], "READY_TO_LAND")
        self.assertEqual(result["trace"][-1]["position"], (25, 4))
        rejection = next(frame for frame in result["trace"] if frame["state"] == "LANDING_REJECTED")
        self.assertEqual(rejection["landing"]["surface"], "water")
        self.assertEqual(result["trace"][-1]["landing"]["source"], "rgbd_plane_fit")

    def test_invalid_observation_types_stop_instead_of_crashing(self):
        class Sensor:
            def __init__(self, observation): self.observation = observation
            def observe(self, scenario, position, tick): return self.observation
        for obs in (Observation(0, frozenset(), None), Observation(0, frozenset(), True),
                    Observation(False, frozenset(), 1.0), Observation(0, frozenset({(-1, 2)}), 1.0),
                    Observation(0, frozenset(), 1.0, valid="yes")):
            result = run_mission(vision_scenarios()[0], Sensor(obs))
            self.assertEqual(result["metrics"]["terminal_state"], "SENSOR_HOLD")
            self.assertEqual(result["metrics"]["grid_moves"], 0)


if __name__ == "__main__":
    unittest.main()
