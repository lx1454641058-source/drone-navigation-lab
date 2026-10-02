"""来源：本项目原创。只读重放保存的相机帧，核验规划日志与文件摘要。"""

import argparse
import hashlib
import json
from pathlib import Path

from .exploration import explore
from .pinhole import Intrinsics, PerspectiveFrame, Pose


class RecordedCamera:
    def __init__(self, frames):
        self.frames, self.index = frames, 0

    def capture(self, pose, intrinsics, tick):
        if self.index >= len(self.frames):
            raise ValueError("recorded frames exhausted")
        item = self.frames[self.index]
        self.index += 1
        saved_pose = Pose(**{key: tuple(value) for key, value in item['pose'].items()})
        saved_k = Intrinsics(**item['intrinsics'])
        if pose != saved_pose or intrinsics != saved_k or tick != item['tick']:
            raise ValueError("requested observation differs from recorded pose/time/calibration")
        return PerspectiveFrame(saved_k, saved_pose, tuple(tuple(rgb) for rgb in item['rgb']),
                                tuple(item['depth_z_m']), item['tick'])


def verify(output: Path) -> dict:
    report = json.loads((output/'exploration_report.json').read_text(encoding='utf-8'))
    manifest = json.loads((output/'observations_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest:
        path = (output/entry['path']).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("manifest path escapes experiment directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError("observation digest mismatch")
    checked = []
    for saved in report['results']:
        name = saved['key']+'-observations.json'
        entries = [entry for entry in manifest if entry['path'] == name]
        if len(entries) != 1:
            raise ValueError("observation file must appear exactly once in verified manifest")
        frames = json.loads((output/name).read_text(encoding='utf-8'))
        if len(frames) != entries[0]['frames']:
            raise ValueError("observation frame count differs from manifest")
        camera = RecordedCamera(frames)
        replay = explore(camera, saved['width'], saved['height'], tuple(saved['start']), tuple(saved['goal']),
                         max_ticks=saved['max_ticks'], previews=False)
        # JSON 规范化把内存中的 tuple 与文件中的 array 对齐；画面编码不是规划输入。
        replay = json.loads(json.dumps(replay, ensure_ascii=False, allow_nan=False))
        target = {key:saved[key] for key in replay}
        target = json.loads(json.dumps(target))
        for frame in target['trace']:
            frame['vision'] = None
        if replay != target or camera.index != len(frames):
            raise ValueError("replayed mission differs from saved result: "+saved['key'])
        checked.append({'scenario':saved['key'], 'frames':camera.index, 'terminal':replay['terminal_state']})
    return {'verified':True, 'results':checked}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='只读重放相机记录并核验探索结果')
    parser.add_argument('output', type=Path)
    print(json.dumps(verify(parser.parse_args().output), ensure_ascii=False))
