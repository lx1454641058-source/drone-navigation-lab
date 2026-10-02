"""来源：本项目原创。根据原始相机帧重算运动、地图和探索决策。"""

import argparse
import hashlib
import json
from pathlib import Path

from .motion import MotionConfig
from .timed_navigation import navigate
from .verify_exploration import RecordedCamera


def verify(output: Path):
    report = json.loads((output/'motion_report.json').read_text(encoding='utf-8'))
    manifest = json.loads((output/'observations_manifest.json').read_text(encoding='utf-8'))
    verified_files = {}
    for entry in manifest:
        path = (output/entry['path']).resolve()
        if not path.is_relative_to(output.resolve()) or entry['path'] in verified_files:
            raise ValueError('invalid or duplicate manifest path')
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry['sha256']:
            raise ValueError('observation digest mismatch')
        frames = json.loads(blob)
        if len(frames) != entry['frames']:
            raise ValueError('observation frame count mismatch')
        verified_files[entry['path']] = frames
    checked, keys = [], set()
    for saved in report['results']:
        name = saved['key']+'-observations.json'
        if name not in verified_files or saved['key'] in keys:
            raise ValueError('missing observations or duplicate scenario')
        keys.add(saved['key'])
        camera = RecordedCamera(verified_files[name])
        replay = navigate(camera, saved['width'], saved['height'], tuple(saved['start']), tuple(saved['goal']),
                          config=MotionConfig(**saved['config']), max_ticks=saved['max_ticks'],
                          failure_edge=saved['failure_edge'], failure_at_s=saved['failure_at_s'],
                          policy_mode=saved['policy_mode'], previews=False)
        replay = json.loads(json.dumps(replay, allow_nan=False))
        target = json.loads(json.dumps({key:saved[key] for key in replay}))
        for frame in target['trace']:
            frame['vision'] = None
        if replay != target or camera.index != len(camera.frames):
            raise ValueError('replayed navigation differs: '+saved['key'])
        checked.append({'scenario':saved['key'], 'frames':camera.index, 'terminal':replay['terminal_state']})
    if len(verified_files) != len(keys):
        raise ValueError('unexpected observations in manifest')
    return {'verified':True, 'scope':'观测文件完整性及导航输出重算；不认证数据来源或独立评价指标', 'results':checked}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从保存的观测只读重算运动实验')
    parser.add_argument('output', type=Path)
    print(json.dumps(verify(parser.parse_args().output), ensure_ascii=False))
