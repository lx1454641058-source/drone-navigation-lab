"""来源：本项目原创 Gaussian Naive Bayes 实现；算法资料见 docs/SOURCES.md。

高斯朴素贝叶斯从带类别的像素学习均值和方差。它只适合本合成色彩基线，
不理解真实物体形状；模型分数不是已经校准的正确概率。
"""

import hashlib
import json
from dataclasses import dataclass
from math import isfinite, log

from .imaging import CLASSES, RGB


def features(pixel: RGB) -> tuple[float, float, float]:
    total = max(sum(pixel), 1)
    return pixel[0]/total, pixel[1]/total, total/765


@dataclass(frozen=True)
class ColorModel:
    means: tuple[tuple[float, ...], ...]
    variances: tuple[tuple[float, ...], ...]
    counts: tuple[int, ...]
    reject_distance: float = 30.0

    @classmethod
    def fit(cls, images: list[tuple[list[RGB], list[int]]]) -> "ColorModel":
        sums = [[0.0]*3 for _ in CLASSES]
        squares = [[0.0]*3 for _ in CLASSES]
        counts = [0]*len(CLASSES)
        for pixels, labels in images:
            if len(pixels) != len(labels):
                raise ValueError("training pixels and labels differ in length")
            for pixel, label in zip(pixels, labels):
                if type(label) is not int or not 0 <= label < len(CLASSES):
                    raise ValueError("invalid training label")
                if len(pixel) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in pixel):
                    raise ValueError("invalid training pixel")
                counts[label] += 1
                for axis, value in enumerate(features(pixel)):
                    sums[label][axis] += value
                    squares[label][axis] += value*value
        if min(counts) < 2:
            raise ValueError("each class needs at least two training pixels")
        means = tuple(tuple(v/counts[i] for v in row) for i, row in enumerate(sums))
        variances = tuple(tuple(max(0.00005, squares[i][j]/counts[i]-means[i][j]**2)
                                for j in range(3)) for i in range(len(CLASSES)))
        return cls(means, variances, tuple(counts))

    def predict(self, pixel: RGB) -> int:
        f = features(pixel)
        scores = []
        distances = []
        for mean, variance, count in zip(self.means, self.variances, self.counts):
            distance = sum((f[j]-mean[j])**2/variance[j] for j in range(3))
            distances.append(distance)
            scores.append(log(count) - 0.5*(distance + sum(log(v) for v in variance)))
        winner = max(range(len(scores)), key=scores.__getitem__)
        return winner if distances[winner] <= self.reject_distance and sum(pixel) > 45 else -1

    def predict_image(self, pixels: tuple[RGB, ...] | list[RGB]) -> list[int]:
        # 相同颜色复用结果，不跨模型、帧或配置共享缓存。
        cache: dict[RGB, int] = {}
        labels = []
        for pixel in pixels:
            if pixel not in cache:
                cache[pixel] = self.predict(pixel)
            labels.append(cache[pixel])
        return labels

    def to_dict(self) -> dict:
        return {"kind": "gaussian_color_v1", "classes": list(CLASSES), "means": self.means,
                "variances": self.variances, "counts": self.counts,
                "reject_distance": self.reject_distance,
                "scope": "procedural RGB imagery only; not a real-world detector"}

    @classmethod
    def from_dict(cls, data: dict) -> "ColorModel":
        if data.get("kind") != "gaussian_color_v1" or data.get("classes") != list(CLASSES):
            raise ValueError("unsupported model kind/classes")
        means = data.get("means", [])
        variances = data.get("variances", [])
        counts = data.get("counts", [])
        threshold = data.get("reject_distance")
        for rows in (means, variances):
            if len(rows) != len(CLASSES) or any(len(row) != 3 for row in rows):
                raise ValueError("invalid model shape")
            if any(type(v) not in (int, float) or not isfinite(v) for row in rows for v in row):
                raise ValueError("invalid model numeric values")
        if any(v <= 0 for row in variances for v in row):
            raise ValueError("variance must be positive")
        if len(counts) != len(CLASSES) or any(type(v) is not int or v < 2 for v in counts):
            raise ValueError("invalid training counts")
        if type(threshold) not in (int, float) or not isfinite(threshold) or threshold <= 0:
            raise ValueError("invalid rejection threshold")
        return cls(tuple(map(tuple, means)), tuple(map(tuple, variances)), tuple(counts), threshold)

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()
