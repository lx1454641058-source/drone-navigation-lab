"""来源：本项目原创。把原始 PNG 与已验证 RGB 帧绑定后直接存档。"""
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter

from .packed_rgbd import PackedFrame,load_packed,prepare_packed
from .tum_rgbd import safe_path
from .vision_runtime import CPUVisionSession
from .realvision import dump,sha
from tools.tinyformer_probe import MODEL_SHA


@dataclass(frozen=True)
class EncodedFrame:
    frame: PackedFrame
    png_bytes: bytes

    def __post_init__(self):
        from PIL import Image
        if type(self.frame) is not PackedFrame or type(self.png_bytes) is not bytes:
            raise ValueError('immutable validated RGB frame and PNG bytes required')
        k=self.frame.intrinsics
        # A file path alone cannot bind an archive to the frame used by the model:
        # the file could change after load_packed. Decode these exact immutable
        # bytes and compare them before allowing either inference or archival.
        try:
            with Image.open(BytesIO(self.png_bytes)) as image:
                if image.format!='PNG' or image.mode!='RGB' or image.size!=(k.width,k.height):
                    raise ValueError('PNG format, dimensions or colour mode differ')
                if image.tobytes()!=self.frame.rgb_bytes:
                    raise ValueError('PNG pixels differ from model frame')
        except OSError as exc:
            raise ValueError('PNG could not be decoded') from exc


def load_encoded(root,selection,origin_s):
    """Keep all old loading checks; extra read/decode is included in measured time."""
    observation=load_packed(root,selection,origin_s)
    encoded=EncodedFrame(observation.frame,safe_path(root,selection['rgb'][1]).read_bytes())
    return observation,encoded


class EncodedVisionSession(CPUVisionSession):
    def detect_encoded(self,encoded):
        from PIL import Image
        if self.replay or type(encoded) is not EncodedFrame or self.input_mode!='raw':
            raise ValueError('validated encoded frame and raw tensor mode required')
        start=perf_counter(); index=len(self.records)
        directory=self.output/f'frame-{index:04d}'; directory.mkdir(exist_ok=False)
        frame=encoded.frame; k=frame.intrinsics
        image=Image.frombytes('RGB',(k.width,k.height),frame.rgb_bytes)
        with (directory/'input.png').open('xb') as stream: stream.write(encoded.png_bytes)
        png_at=perf_counter()
        window=dict(index=0,x0=0,y0=0,width=k.width,height=k.height)
        tensor=prepare_packed(image,window); prepared=perf_counter()
        with (directory/'input.f32').open('xb') as stream: stream.write(tensor)
        archived=perf_counter()
        result=self._request(dict(directory=str(directory),window=window)); returned=perf_counter()
        record=dict(index=index,directory=directory.name,input_mode='raw',threads=self.threads,
            image_archive_mode='validated-source-png',window=window,
            input_rgb_sha256=sha(directory/'input.png'),model_sha256=MODEL_SHA,
            elapsed_s=returned-start,prepare_s=prepared-start,input_archive_s=archived-prepared,result=result,
            python_stages_s=dict(png=png_at-start,tensor=prepared-png_at,input_archive=archived-prepared,
                                 worker_roundtrip=returned-archived))
        self.records.append(record); dump(directory/'call.json',record)
        return record
