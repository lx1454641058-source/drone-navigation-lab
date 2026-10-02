"""来源：本项目原创。图像契约与无第三方依赖的 PNG 输出。"""

import base64
import struct
import zlib
from dataclasses import dataclass

RGB = tuple[int, int, int]
CLASSES = ("paved", "vegetation", "water", "building", "person")
DISPLAY_COLORS = ((188, 194, 194), (81, 151, 83), (60, 145, 210),
                  (192, 90, 62), (244, 195, 59), (182, 60, 192))


@dataclass(frozen=True)
class RGBDFrame:
    """正交俯视相机：深度为沿竖直向下方向的米数；不含标签或场景对象。"""

    width: int
    height: int
    rgb: tuple[RGB, ...]
    depth_m: tuple[float, ...]
    origin_cell: tuple[int, int]
    pixels_per_cell: int
    camera_altitude_m: float
    tick: int

    def validate(self) -> None:
        if (type(self.width) is not int or type(self.height) is not int
                or min(self.width, self.height) <= 0 or self.width*self.height > 1_000_000):
            raise ValueError("invalid frame dimensions")
        if len(self.rgb) != self.width*self.height or len(self.depth_m) != len(self.rgb):
            raise ValueError("RGB/depth shape mismatch")
        if type(self.pixels_per_cell) is not int or self.pixels_per_cell < 1:
            raise ValueError("invalid pixels_per_cell")
        if self.width % self.pixels_per_cell or self.height % self.pixels_per_cell:
            raise ValueError("frame must align with grid cells")
        if len(self.origin_cell) != 2 or any(type(v) is not int for v in self.origin_cell):
            raise ValueError("invalid grid origin")
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("invalid frame time")
        from math import isfinite
        if (not isinstance(self.camera_altitude_m, (int, float))
                or isinstance(self.camera_altitude_m, bool)
                or not isfinite(self.camera_altitude_m) or self.camera_altitude_m <= 0):
            raise ValueError("invalid camera altitude")
        if any(len(pixel) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in pixel)
               for pixel in self.rgb):
            raise ValueError("RGB pixels must contain three uint8 values")


def png_bytes(width: int, height: int, pixels: tuple[RGB, ...] | list[RGB]) -> bytes:
    if width <= 0 or height <= 0 or len(pixels) != width*height:
        raise ValueError("invalid PNG dimensions")

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack("!I", len(body)) + kind + body + struct.pack("!I", zlib.crc32(kind+body))

    rows = b"".join(b"\0" + bytes(v for pixel in pixels[y*width:(y+1)*width] for v in pixel)
                    for y in range(height))
    header = struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def png_url(width: int, height: int, pixels: tuple[RGB, ...] | list[RGB]) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(width, height, pixels)).decode("ascii")
