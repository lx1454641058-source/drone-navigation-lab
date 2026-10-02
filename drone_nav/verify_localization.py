"""来源：本项目原创。重放带误差标签的深度帧，独立重算真实轨迹与汇总。"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .localization import LocalizationBudget, PositionError, audit_localization
from .localization_experiment import summarize
from .motion import MotionConfig
from .raycast import Box,Surface,World
from .timed_navigation import navigate
from .verify_exploration import RecordedCamera


def verify(output: Path):
    report = json.loads((output/'localization_report.json').read_text(encoding='utf-8'))
    manifest = json.loads((output/'observations_manifest.json').read_text(encoding='utf-8'))
    entries = {e['path']:e for e in manifest}
    if len(entries)!=len(manifest) or len(entries)!=len(report['results']):
        raise ValueError('manifest contains duplicate or unexpected files')
    keys,checked = set(),[]
    for saved in report['results']:
        key = saved['key']
        name = key+'-observations.json.gz'
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()) or name not in entries or key in keys:
            raise ValueError('invalid observation path or duplicated scenario')
        keys.add(key)
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=entries[name]['sha256']:
            raise ValueError('observation digest mismatch')
        frames = json.loads(gzip.decompress(blob))
        if len(frames)!=entries[name]['frames']:
            raise ValueError('observation frame count mismatch')
        camera = RecordedCamera(frames)
        budget = LocalizationBudget(**saved['localization_budget'])
        replay = navigate(camera,saved['width'],saved['height'],tuple(saved['start']),tuple(saved['goal']),
                          config=MotionConfig(**saved['config']),max_ticks=saved['max_ticks'],
                          localization=budget,policy_mode=saved['policy_mode'],previews=False)
        replay = json.loads(json.dumps(replay,allow_nan=False))
        target = json.loads(json.dumps({key:saved[key] for key in replay}))
        for frame in target['trace']:
            frame['vision'] = None
        if replay!=target or camera.index!=len(frames):
            raise ValueError('navigation replay mismatch: '+key)
        world_data = saved['world']
        surfaces = tuple(Box(**s) if 'low' in s else Surface(**s) for s in world_data['surfaces'])
        world = World(surfaces,world_data['width_m'],world_data['height_m'])
        audit = audit_localization(world,replay,PositionError(**saved['error_model']),budget)
        if json.loads(json.dumps(audit))!=saved['audit']:
            raise ValueError('independent geometry audit mismatch: '+key)
        checked.append({'key':key,'frames':camera.index,'terminal':replay['terminal_state']})
    if summarize(report['results'])!=report['summary']:
        raise ValueError('summary differs from individual results')
    return {'verified':True,'results':checked,
            'scope':'已保存观测的决策重放及给定误差模型的真值轨迹重算；未认证传感器真实性。'}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='重放定位误差实验')
    parser.add_argument('output',type=Path)
    print(json.dumps(verify(parser.parse_args().output),ensure_ascii=False))
