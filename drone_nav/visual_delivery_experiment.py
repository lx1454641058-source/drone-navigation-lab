"""来源：本项目原创。导航—降落检查开发场景；几何真值仅用于相机渲染。"""

import gzip
import hashlib
import json
from dataclasses import asdict
from math import radians,tan
from pathlib import Path

from .exploration_experiment import SimulatedDepthCamera
from .pinhole import Intrinsics
from .raycast import Box,Surface,World,render
from .vision_experiment import train_model
from .visual_delivery import run_visual_delivery


class VisualDeliveryCamera(SimulatedDepthCamera):
    def __init__(self,world,*,light=1.0,dropout=0.0,record=False):
        super().__init__(world,record=record)
        self.light,self.dropout,self.landing_frames = light,dropout,[]

    def capture_landing(self,pose,intrinsics,tick):
        frame,_ = render(self.world,pose,intrinsics,tick=tick,seed=1601,
                         light=self.light,dropout=self.dropout)
        if self.record:
            self.landing_frames.append(asdict(frame))
        return frame


def delivery_cases():
    def scene(key,title,label=0,angle=0,objects=(),expected='LANDING_REJECTED',**settings):
        slope = tan(radians(angle))
        ground = Surface('ground',label,(-40,60,-40,56),z0=-slope*16.5,slope_x=slope)
        return dict(key=key,title=title,world=World((ground,*objects)),expected=expected,
                    light=1.0,dropout=0.0,**settings)
    cases = [
        scene('paved','平坦铺装取餐点',expected='READY_TO_LAND'),
        scene('water','平坦水面',label=2),
        scene('vegetation','平坦植被',label=1),
        scene('steep','8° 铺装斜坡',angle=8),
        scene('person','取餐点有人',objects=(Box('person',4,(16.2,8.2,0),(16.8,8.8,1.6)),)),
        scene('bump','同色 25 cm 凸起',objects=(Box('bump',0,(16.1,8.1,0),(16.9,8.9,.25)),)),
        scene('unknown','未知材质',label=-1),
        scene('dark','检查时弱光'),
        scene('missing','检查时深度缺失'),
        scene('narrow','相机视野不足'),
        scene('gentle','3° 铺装缓坡',angle=3,expected='READY_TO_LAND'),
        scene('wall','隔墙无法到达',objects=(Box('wall',3,(8,0,0),(10,16,6)),),expected='NAVIGATION_HOLD'),
    ]
    cases[7]['light'] = .08
    cases[8]['dropout'] = .03
    cases[9]['intrinsics'] = Intrinsics(96,72,80,80,47.5,35.5)
    return cases


def summarize(results):
    inspected = [r for r in results if r['landing'] is not None]
    return {'cases':len(results),'inspected':len(inspected),
            'ready':sum(r['terminal_state']=='READY_TO_LAND' for r in results),
            'rejected':sum(r['terminal_state']=='LANDING_REJECTED' for r in results),
            'navigation_hold':sum(r['terminal_state']=='NAVIGATION_HOLD' for r in results),
            'sensor_hold':sum(r['terminal_state']=='LANDING_SENSOR_HOLD' for r in results),
            'geometry_only_pass':sum(r['landing']['geometry_accepted'] for r in inspected),
            'expected_matches':sum(r['terminal_state']==r['expected'] for r in results)}


def run_delivery_suite(output: Path):
    model = train_model()
    manifest,results = [],[]

    def save(name,blob,**counts):
        (output/name).write_bytes(blob)
        manifest.append(dict(path=name,sha256=hashlib.sha256(blob).hexdigest(),**counts))

    save('model.json',json.dumps(model.to_dict(),ensure_ascii=False).encode('utf-8'))
    for case in delivery_cases():
        camera = VisualDeliveryCamera(case['world'],light=case['light'],dropout=case['dropout'],record=True)
        result = run_visual_delivery(camera,model,(3,8),(16,8),landing_intrinsics=case.get('intrinsics'))
        result.update(key=case['key'],title=case['title'],expected=case['expected'],
                      world=asdict(case['world']),camera_settings=dict(light=case['light'],dropout=case['dropout'],
                                                                     navigation_seed=1201,landing_seed=1601))
        blob = json.dumps(dict(navigation=camera.frames,landing=camera.landing_frames),
                          separators=(',',':'),allow_nan=False).encode('utf-8')
        save(case['key']+'-observations.json.gz',gzip.compress(blob,mtime=0),
             navigation_frames=len(camera.frames),landing_frames=len(camera.landing_frames))
        results.append(result)
        print(case['key'],result['terminal_state'],result['reason'],flush=True)
    report = dict(version='0.8.0',mode='delivery',summary=summarize(results),results=results,
                  scope='固定开发场景，不是独立测试集；精确位姿、合成颜色与理想深度；未模拟下降、触地、放餐。')
    (output/'delivery_report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (output/'observations_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    template = (Path(__file__).parent/'delivery_lab.html').read_text(encoding='utf-8')
    (output/'demo.html').write_text(template.replace('__DELIVERY_DATA__',json.dumps(
        report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')),encoding='utf-8')
    return report
