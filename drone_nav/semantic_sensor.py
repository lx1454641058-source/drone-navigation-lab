"""来源：本项目原创。真实预训练检测器的本地相机适配，空检测不证明空闲。"""
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import tempfile
from threading import Thread
from time import perf_counter

from .detection_bridge import DetectionPacket, SpatialContext, BridgeConfig, project_detections
from .realvision import asset_root, validate_assets, sha, dump
from tools.tinyformer_probe import MODEL, MODEL_SHA, prepare

ROOT = Path(__file__).resolve().parents[1]


class TinyFormerSession:
    """Model loads before a simulated vehicle is allocated; calls have a finite timeout."""
    def __init__(self, output, *, replay=False, timeout_s=30):
        self.output = Path(output).resolve()
        self.replay = replay
        self.timeout_s = timeout_s
        self.records = []
        self.process = None
        self.stderr = None
        if not replay:
            if sha(MODEL) != MODEL_SHA:
                raise ValueError('TinyFormer model differs')
            runtime = asset_root().resolve()
            _, lock = validate_assets(runtime)
            self.output.mkdir(parents=True, exist_ok=False)
            dump(self.output / 'runtime-lock.json', lock)
        else:
            runtime = asset_root().resolve()
        self.stderr = (tempfile.TemporaryFile(mode='w+', encoding='utf-8') if replay
                       else (self.output / 'stderr.txt').open('x', encoding='utf-8'))
        self.messages = Queue()
        start = perf_counter()
        try:
            self.process = subprocess.Popen(['node', str(ROOT / 'tools/semantic_sensor_worker.cjs')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                text=True, encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self.reader = Thread(target=self._read, daemon=True)
            self.reader.start()
            response = self._request(dict(model=str(MODEL), runtime=str(runtime), replay=replay))
            if response != {'ready': True}:
                raise ValueError('detector did not initialize')
            self.startup_s = perf_counter() - start
        except BaseException:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            self.messages.put(line)
        self.messages.put(None)

    def _request(self, request):
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        try:
            line = self.messages.get(timeout=self.timeout_s)
        except Empty:
            self.process.kill()
            self.process.wait()
            raise ValueError('detector timeout; no visual authorization') from None
        if line is None:
            raise ValueError('detector exited without a result')
        result = json.loads(line)
        if 'error' in result:
            raise ValueError('detector: ' + result['error'])
        return result

    def detect(self, frame):
        from PIL import Image
        if self.replay:
            raise ValueError('use replay_frame for saved scores')
        frame.validate()
        started = perf_counter()
        index = len(self.records)
        directory = self.output / f'frame-{index:04d}'
        directory.mkdir(exist_ok=False)
        k = frame.intrinsics
        image = Image.new('RGB', (k.width, k.height))
        image.putdata(frame.rgb)
        image.save(directory / 'input.png')
        window = dict(index=0, x0=0, y0=0, width=k.width, height=k.height)
        tensor = prepare(image, window)
        prepared = perf_counter()
        # Compression level changes file size/cost, never the float32 tensor.
        (directory / 'input.gz').write_bytes(gzip.compress(tensor, compresslevel=1, mtime=0))
        archived = perf_counter()
        result = self._request(dict(directory=str(directory), window=window))
        elapsed = perf_counter() - started
        record = dict(index=index, directory=directory.name, window=window,
                      input_rgb_sha256=sha(directory / 'input.png'), elapsed_s=elapsed,
                      prepare_s=prepared-started, input_archive_s=archived-prepared,
                      model_sha256=MODEL_SHA, result=result)
        self.records.append(record)
        dump(directory / 'call.json', record)
        return record

    def replay_frame(self, record):
        if not self.replay:
            raise ValueError('replay session required')
        directory = self.output / record['directory']
        return self._request(dict(directory=str(directory), window=record['window']))

    def close(self):
        if self.process is not None:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.reader.join(timeout=1)
            self.process.stdout.close()
        if self.stderr is not None:
            self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def packet_for(record, frame, *, frame_id, captured_at_s, completed_at_s):
    boxes = record['result']['boxes']
    detections = tuple(dict(id=i, group=b['group'], score=b['score'],
        box=[b[n] for n in ('x1', 'y1', 'x2', 'y2')]) for i, b in enumerate(boxes))
    packet = DetectionPacket(frame_id, 'simulation-rgbd', frame.intrinsics.width,
        frame.intrinsics.height, detections, 'TinyFormer:' + record['model_sha256'],
        captured_at_s, completed_at_s, 'physics-seconds')
    packet.validate()
    return packet


def context_for(frame, frame_id, captured_at_s):
    return SpatialContext(frame_id, 'simulation-rgbd', 'physics-seconds', captured_at_s,
        captured_at_s, frame.intrinsics, frame.pose, frame.depth_z_m, True,
        'camera_optical_axis_z_m', 'actual_simulated_range_capture',
        'exact_simulated_capture_pose', 'simulation-pinhole')


def visual_move_decision(packet, context, *, now_s, max_age_s=.5):
    """A veto before the original geometric gate, never a replacement flight authorization.

    A recognized person/vehicle has unknown extent even when samples project;
    stop for review rather than treating its box as a complete safe boundary.
    """
    projection = project_detections(packet, context, now_s=now_s,
        now_clock_id='physics-seconds', config=BridgeConfig(max_age_s=max_age_s))
    if projection['status'] == 'CONTEXT_REJECTED':
        reason = 'VISUAL_CONTEXT_HOLD'
    elif packet.detections:
        reason = 'VISUAL_TARGET_HOLD'
    else:
        reason = 'NO_DETECTION_REQUIRES_GEOMETRY'
    return dict(permit_geometry_check=reason == 'NO_DETECTION_REQUIRES_GEOMETRY',
                reason=reason, projection=projection, flight_authorized=False)
