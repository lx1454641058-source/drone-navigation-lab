"""来源：本项目原创。实验性 CPU 视觉运行器，复用原进程协议和输入契约。"""
import gzip
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import tempfile
from threading import Thread
from time import perf_counter

from .packed_rgbd import PackedFrame,prepare_packed
from .realvision import asset_root,validate_assets,sha,dump
from .semantic_sensor import TinyFormerSession
from tools.tinyformer_probe import MODEL,MODEL_SHA

ROOT=Path(__file__).resolve().parents[1]


class CPUVisionSession(TinyFormerSession):
    """One sequential model session; changing options never mutates the old default."""
    def __init__(self,output,*,threads=8,input_mode='raw',replay=False,timeout_s=30):
        if type(threads) is not int or threads not in (4,8,16) or input_mode not in ('gzip','raw'):
            raise ValueError('unsupported CPU runtime configuration')
        if type(timeout_s) not in (int,float) or not 0<timeout_s<=120:
            raise ValueError('invalid runtime timeout')
        self.output=Path(output).resolve(); self.threads=threads; self.input_mode=input_mode
        self.replay=replay; self.timeout_s=timeout_s; self.records=[]; self.process=None; self.stderr=None
        runtime=asset_root().resolve()
        if not replay:
            if sha(MODEL)!=MODEL_SHA: raise ValueError('TinyFormer model differs')
            _,lock=validate_assets(runtime)
            self.output.mkdir(parents=True,exist_ok=False)
            dump(self.output/'runtime-lock.json',lock)
            dump(self.output/'configuration.json',dict(threads=threads,input_mode=input_mode,
                worker_sha256=sha(ROOT/'tools/vision_runtime_worker.cjs'),model_sha256=MODEL_SHA))
        else:
            config=json.loads((self.output/'configuration.json').read_text(encoding='utf-8'))
            expected=dict(threads=threads,input_mode=input_mode,
                worker_sha256=sha(ROOT/'tools/vision_runtime_worker.cjs'),model_sha256=MODEL_SHA)
            if config!=expected: raise ValueError('saved runtime configuration differs')
        self.stderr=(tempfile.TemporaryFile(mode='w+',encoding='utf-8') if replay
            else (self.output/'stderr.txt').open('x',encoding='utf-8'))
        self.messages=Queue(); start=perf_counter()
        try:
            self.process=subprocess.Popen(['node',str(ROOT/'tools/vision_runtime_worker.cjs')],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding='utf-8',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.reader=Thread(target=self._read,daemon=True); self.reader.start()
            answer=self._request(dict(model=str(MODEL),runtime=str(runtime),replay=replay,
                threads=threads,input_mode=input_mode))
            if answer!=dict(ready=True,threads=threads,input_mode=input_mode):
                raise ValueError('worker configuration not acknowledged')
            self.startup_s=perf_counter()-start
        except BaseException:
            self.close(); raise

    def detect_packed(self,frame):
        from PIL import Image
        if self.replay or type(frame) is not PackedFrame:
            raise ValueError('validated immutable packed frame required')
        start=perf_counter(); index=len(self.records)
        directory=self.output/f'frame-{index:04d}'; directory.mkdir(exist_ok=False)
        k=frame.intrinsics; image=Image.frombytes('RGB',(k.width,k.height),frame.rgb_bytes)
        image.save(directory/'input.png'); png_at=perf_counter()
        window=dict(index=0,x0=0,y0=0,width=k.width,height=k.height)
        tensor=prepare_packed(image,window); prepared=perf_counter()
        name='input.gz' if self.input_mode=='gzip' else 'input.f32'
        payload=gzip.compress(tensor,compresslevel=1,mtime=0) if self.input_mode=='gzip' else tensor
        with (directory/name).open('xb') as stream: stream.write(payload)
        archived=perf_counter()
        result=self._request(dict(directory=str(directory),window=window)); returned=perf_counter()
        record=dict(index=index,directory=directory.name,window=window,input_mode=self.input_mode,
            threads=self.threads,input_rgb_sha256=sha(directory/'input.png'),model_sha256=MODEL_SHA,
            elapsed_s=returned-start,prepare_s=prepared-start,input_archive_s=archived-prepared,result=result,
            python_stages_s=dict(png=png_at-start,tensor=prepared-png_at,input_archive=archived-prepared,
                                 worker_roundtrip=returned-archived))
        self.records.append(record); dump(directory/'call.json',record)
        return record
