"""来源：本项目原创。只用保存的模型和 RGB-D 原始帧重算两阶段决策。"""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

from .classifier import ColorModel
from .perspective_landing import LandingConfig
from .pinhole import Intrinsics
from .verify_exploration import RecordedCamera
from .visual_delivery import run_visual_delivery
from .visual_delivery_experiment import summarize


class RecordedDeliveryCamera(RecordedCamera):
    def __init__(self,bundle):
        super().__init__(bundle['navigation'])
        self.landing_camera = RecordedCamera(bundle['landing'])

    def capture_landing(self,pose,intrinsics,tick):
        return self.landing_camera.capture(pose,intrinsics,tick)


def normalized(result):
    item = copy.deepcopy(result)
    item['landing_images'] = None
    for step in item['navigation']['trace']:
        step['vision'] = None
    return json.loads(json.dumps(item,allow_nan=False))


def verify(output: Path):
    report = json.loads((output/'delivery_report.json').read_text(encoding='utf-8'))
    manifest = json.loads((output/'observations_manifest.json').read_text(encoding='utf-8'))
    entries = {entry['path']:entry for entry in manifest}
    required = {'model.json',*(r['key']+'-observations.json.gz' for r in report['results'])}
    if len(entries)!=len(manifest) or set(entries)!=required or len(report['results'])!=len(required)-1:
        raise ValueError('manifest must contain each model/observation file exactly once')
    for name,entry in entries.items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError('manifest path escapes experiment directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('digest mismatch: '+name)
    model = ColorModel.from_dict(json.loads((output/'model.json').read_text(encoding='utf-8')))
    checked = []
    for saved in report['results']:
        name = saved['key']+'-observations.json.gz'
        bundle = json.loads(gzip.decompress((output/name).read_bytes()))
        entry = entries[name]
        if len(bundle['navigation'])!=entry['navigation_frames'] or len(bundle['landing'])!=entry['landing_frames']:
            raise ValueError('frame count mismatch')
        camera = RecordedDeliveryCamera(bundle)
        nav = saved['navigation']
        replay = run_visual_delivery(camera,model,tuple(nav['start']),tuple(nav['goal']),
                                     reference_z_m=saved['reference_z_m'],landing_config=LandingConfig(**saved['landing_config']),
                                     landing_intrinsics=Intrinsics(**saved['landing_intrinsics']),previews=False)
        target = {key:saved[key] for key in replay}
        if normalized(replay)!=normalized(target) or camera.index!=len(bundle['navigation']) or camera.landing_camera.index!=len(bundle['landing']):
            raise ValueError('replayed decision differs: '+saved['key'])
        checked.append(dict(key=saved['key'],navigation_frames=camera.index,
                            landing_frames=camera.landing_camera.index,terminal=replay['terminal_state']))
    if summarize(report['results'])!=report['summary']:
        raise ValueError('summary differs')
    return dict(verified=True,results=checked,scope='验证可重复性，不证明场景真实性或飞行安全')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='重放导航与降落检查观测')
    parser.add_argument('output',type=Path)
    print(json.dumps(verify(parser.parse_args().output),ensure_ascii=False))
