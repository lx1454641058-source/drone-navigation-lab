"""来源：本项目原创。相机渲染与图像推理之间的适配层。"""

from .classifier import ColorModel
from .mission import Observation, Scenario
from .perception import analyze_frame, frame_preview, landing_evidence
from .rendering import OrthographicCamera


class ImageSensor:
    source = "procedural_rgbd_gaussian_color_v1"
    use_terrain_prior = False
    limitations = ("合成正交俯视 RGB-D 图像与颜色学习基线；没有真实照片、深度学习模型、"
                   "透视遮挡或飞行动力学；距离来自模拟深度通道，不是由单张 RGB 估计")

    def __init__(self, scenario: Scenario, model: ColorModel, *, semantic: bool = True,
                 preview: bool = True, seed: int = 700):
        self.camera = OrthographicCamera(scenario, seed=seed)
        self.model = model
        self.semantic = semantic
        self.preview_enabled = preview
        self.preview = None
        self.landing = None
        self.frame = None
        self.analysis = None

    def observe(self, scenario: Scenario, position: tuple[int, int], tick: int) -> Observation:
        # 只有相机渲染器读取真值。推理仅收到像素、标定、地图边界与离地间距约束。
        self.preview = None
        self.landing = None
        self.frame = self.camera.capture(position, tick)
        if self.frame.tick != tick:
            raise ValueError("camera frame is stale or from the future")
        self.analysis = analyze_frame(self.frame, self.model, (scenario.width, scenario.height),
                                      scenario.min_ground_clearance_m)
        self.preview = frame_preview(self.frame, self.analysis) if self.preview_enabled else None
        quality = min(self.analysis.valid_fraction, self.analysis.exposed_fraction)
        return Observation(tick, self.analysis.obstacles, quality, source=self.source)

    def assess_landing(self, goal: tuple[int, int]) -> dict:
        if self.frame is None or self.analysis is None:
            return {"accepted": False, "reason": "尚未收到图像"}
        self.landing = landing_evidence(self.frame, self.analysis, goal, self.semantic)
        return self.landing
