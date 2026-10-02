"""来源：本项目原创。独立物理台架实验，不宣称视觉导航与飞控已闭环。"""

import argparse
import hashlib
import json
from math import acos,degrees,radians,sqrt
from pathlib import Path

from .flight_control import GRAVITY,MASS,control,euler_quaternion
from .native_physics import DEFAULT_DLL,MODEL,QuadrotorPhysics


def physics_cases():
    return [dict(key='freefall',title='自由落体解析核验',duration_s=.2,mode='off',target=(0,0,1)),
            dict(key='trim_hover',title='重力与四桨推力平衡',duration_s=5,mode='trim',target=(0,0,1)),
            dict(key='altitude',title='高度从 1 m 跟踪到 2 m',duration_s=8,mode='control',target=(0,0,2)),
            dict(key='translate',title='倾斜机体并平移到航点',duration_s=10,mode='control',target=(2,1,1.5)),
            dict(key='tilt',title='初始倾斜 10° 后恢复',duration_s=8,mode='control',target=(0,0,1),roll_deg=10),
            dict(key='loss',title='全部推力失效后的坠落',duration_s=5,mode='loss',target=(0,0,1)),
            dict(key='descent',title='脚本下降与接地停桨',duration_s=12,mode='descent',target=(0,0,.06))]


def run_case(case,dll=DEFAULT_DLL):
    trace,commands = [],[]
    saturated_steps = 0
    stopped_at = None
    with QuadrotorPhysics(dll) as physics:
        physics.reset(quaternion=euler_quaternion(radians(case.get('roll_deg',0)),0))
        trace.append(dict(**physics.state(),ground_distance_m=physics.ground_distance(),rotor_thrusts_n=[0]*4))
        steps = round(case['duration_s']/physics.dt)
        for i in range(steps):
            state = physics.state()
            mode = case['mode']
            target = case['target']
            if mode=='off' or (mode=='loss' and i*physics.dt>=2):
                motors = [0.0]*4
            elif mode in ('trim','loss'):
                motors = [MASS*GRAVITY/4]*4
            else:
                if mode=='descent':
                    target = (0,0,max(.06,1-.15*max(0,i*physics.dt-1)))
                    # 台架脚本使用真实状态判断接地；尚未连接视觉，不用于送餐成功判断。
                    if state['position'][2]<.065 and abs(state['velocity'][2])<.05 and stopped_at is None:
                        stopped_at = state['time_s']
                if stopped_at is not None:
                    motors = [0.0]*4
                else:
                    motors,info = control(state,target)
                    saturated_steps += int(info['saturated'])
            commands.append(motors)
            physics.step(motors)
            if (i+1)%10==0 or i+1==steps:
                trace.append(dict(**physics.state(),ground_distance_m=physics.ground_distance(),rotor_thrusts_n=motors))
        final = trace[-1]
        final_error = sqrt(sum((a-b)**2 for a,b in zip(final['position'],case['target'])))
        tilt = [degrees(acos(max(-1,min(1,1-2*(s['quaternion'][1]**2+s['quaternion'][2]**2))))) for s in trace]
        metrics = dict(final_position_error_m=final_error,final_speed_m_s=sqrt(sum(v*v for v in final['velocity'])),
                       final_ground_distance_m=final['ground_distance_m'],max_tilt_deg=max(tilt),
                       min_ground_distance_m=min(s['ground_distance_m'] for s in trace),
                       saturated_steps=saturated_steps,stopped_at_s=stopped_at,
                       final_tilt_deg=tilt[-1],recorded_samples=len(trace))
        # 解析公式仅用于评价，从不写入仿真位置。
        if case['key']=='freefall':
            t=case['duration_s']
            metrics['analytic_position_error_m']=abs(final['position'][2]-(1-.5*GRAVITY*t*t))
            metrics['analytic_velocity_error_m_s']=abs(final['velocity'][2]+GRAVITY*t)
            passed=max(metrics['analytic_position_error_m'],metrics['analytic_velocity_error_m_s'])<1e-8
        elif case['key']=='trim_hover': passed=final_error<1e-8 and metrics['max_tilt_deg']<1e-6
        elif case['key'] in ('loss','descent'):
            passed=abs(final['ground_distance_m'])<.002 and metrics['final_speed_m_s']<.02
            if case['key']=='descent': passed=passed and stopped_at is not None
        else: passed=final_error<.05 and metrics['final_speed_m_s']<.05 and metrics['final_tilt_deg']<1
        return dict(config=case,engine_version=physics.version,dll_sha256=physics.dll_sha256,
                    model_sha256=physics.model_sha256,dt_s=physics.dt,commands=commands,trace=trace,
                    metrics=metrics,check_passed=passed)


def run_suite(output,dll=DEFAULT_DLL):
    output.mkdir(parents=True,exist_ok=False)
    results=[run_case(c,dll) for c in physics_cases()]
    report=dict(version='0.9.0',mode='physics',results=results,
                scope='独立理想机体台架，使用真实仿真状态反馈；尚未接入视觉导航或真实飞控，不是完整配送。')
    (output/'physics_report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (output/'quadrotor.xml').write_bytes(MODEL.read_bytes())
    template=Path(__file__).with_name('physics_lab.html').read_text(encoding='utf-8')
    (output/'demo.html').write_text(template.replace('__PHYSICS_DATA__',json.dumps(
        report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')),encoding='utf-8')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='独立四旋翼动力学开发实验')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dll',type=Path,default=DEFAULT_DLL)
    args=parser.parse_args()
    report=run_suite(args.output,args.dll)
    for r in report['results']: print(r['config']['key'],r['check_passed'],r['metrics'])
    print('Replay:',(args.output/'demo.html').resolve())
