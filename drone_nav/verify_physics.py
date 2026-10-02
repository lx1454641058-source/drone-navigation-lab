"""来源：本项目原创。核验原始推力日志与物理状态，另重跑反馈控制检查结果。"""

import argparse
import hashlib
import json
from math import radians
from pathlib import Path

from .flight_control import euler_quaternion
from .native_physics import DEFAULT_DLL,MODEL,QuadrotorPhysics
from .physics_experiment import physics_cases,run_case


def verify(output,dll=DEFAULT_DLL):
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if set(manifest)!={'physics_report.json','quadrotor.xml','demo.html'}:
        raise ValueError('unexpected physics manifest')
    for name,digest in manifest.items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('physics artifact digest mismatch: '+name)
    if (output/'quadrotor.xml').read_bytes()!=MODEL.read_bytes():
        raise ValueError('saved model differs from the current validated model')
    report=json.loads((output/'physics_report.json').read_text(encoding='utf-8'))
    if [r['config']['key'] for r in report['results']]!=[c['key'] for c in physics_cases()]:
        raise ValueError('incomplete or reordered experiment matrix')
    checked=[]
    for saved in report['results']:
        rebuilt=json.loads(json.dumps(run_case(saved['config'],dll),allow_nan=False))
        if rebuilt!=saved:raise ValueError('feedback rerun differs: '+saved['config']['key'])
        with QuadrotorPhysics(dll) as q:
            q.reset(quaternion=euler_quaternion(radians(saved['config'].get('roll_deg',0)),0))
            index=0
            for step,motors in enumerate(saved['commands'],1):
                q.step(motors)
                if step%10==0 or step==len(saved['commands']):
                    index+=1
                    replay=dict(**q.state(),ground_distance_m=q.ground_distance(),rotor_thrusts_n=motors)
                    if replay!=saved['trace'][index]:raise ValueError('command replay differs')
            if index+1!=len(saved['trace']):raise ValueError('unconsumed trajectory samples')
        checked.append(dict(key=saved['config']['key'],steps=len(saved['commands']),samples=len(saved['trace'])))
    return dict(verified=True,results=checked,scope='固定引擎/平台可重复性，不是视觉或实机安全证明')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='重放四桨推力记录并核验物理实验')
    parser.add_argument('output',type=Path)
    parser.add_argument('--dll',type=Path,default=DEFAULT_DLL)
    args=parser.parse_args()
    print(json.dumps(verify(args.output,args.dll),ensure_ascii=False))
