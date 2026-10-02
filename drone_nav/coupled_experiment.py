"""来源：本项目原创。物理位姿驱动相机、观测规划和动作回执的联调实验。"""

import gzip,hashlib,json
from dataclasses import asdict
from pathlib import Path

from .native_physics import DEFAULT_DLL
from .physical_navigation import navigate_physical
from .physical_vehicle import PhysicalVehicle,world_xml
from .raycast import Box,Surface,World
from .vision_experiment import train_model


def coupled_cases():
    ground=Surface('ground',0,(-40,60,-40,56))
    building=Box('building',3,(8,6,0),(10,10,6))
    return [dict(key='open',title='开阔场地：真实到达回执',world=World((ground,)),start=(3,8),goal=(8,8)),
            dict(key='building',title='观察建筑后物理绕行',world=World((ground,building)),start=(3,8),goal=(16,8)),
            dict(key='water',title='物理到达后拒绝水面',world=World((Surface('water',2,ground.bounds),)),start=(3,8),goal=(8,8)),
            dict(key='dropout',title='第二段之后深度失效',world=World((ground,building)),start=(3,8),goal=(16,8),dropout_tick=2),
            dict(key='ignored',title='控制器未执行首个移动指令',world=World((ground,)),start=(3,8),goal=(8,8),ignore_move=0),
            dict(key='wall',title='隔墙探索与观察预算',world=World((ground,Box('wall',3,(8,0,0),(10,16,6)))),start=(3,8),goal=(16,8))]


def run_suite(output,dll=DEFAULT_DLL):
    output.mkdir(parents=True,exist_ok=False)
    model=train_model();results=[];manifest=[]
    def save(name,data):
        (output/name).write_bytes(data)
        manifest.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest()))
    save('model.json',json.dumps(model.to_dict()).encode('utf-8'))
    for config in coupled_cases():
        vehicle=PhysicalVehicle(config['world'],config['start'],dll=dll,
            dropout_tick=config.get('dropout_tick'),ignore_move=config.get('ignore_move'))
        try:
            r=navigate_physical(vehicle,model,config['start'],config['goal'])
            if vehicle.history[-1]['time_s']!=vehicle.state()['time_s']:vehicle.history.append(vehicle.state())
            r.update(key=config['key'],title=config['title'],world=asdict(config['world']),
                     dropout_tick=config.get('dropout_tick'),ignore_move=config.get('ignore_move'),
                     engine_version=vehicle.physics.version,dll_sha256=vehicle.physics.dll_sha256,
                     physics_model_sha256=vehicle.physics.model_sha256,physics_history=vehicle.history,
                     actions=vehicle.actions,audit=vehicle.audit,physics_steps=vehicle.steps,
                     camera_frames=len(vehicle.frames))
            blob=json.dumps(dict(frames=vehicle.frames,commands=vehicle.commands),allow_nan=False,separators=(',',':')).encode('utf-8')
            save(config['key']+'-raw.json.gz',gzip.compress(blob,mtime=0))
            save(config['key']+'-world.xml',world_xml(config['world']))
            results.append(r)
            print(config['key'],r['terminal_state'],round(r['elapsed_s'],3),len(vehicle.actions),vehicle.audit,flush=True)
        finally:vehicle.close()
    report=dict(version='0.10.0',mode='coupled',results=results,
                scope='理想深度/颜色、精确仿真位姿和原创控制器的联调；已初始化在 3.5 m 高度，未起飞、下降、放餐。')
    save('coupled_report.json',json.dumps(report,ensure_ascii=False,allow_nan=False).encode('utf-8'))
    template=Path(__file__).with_name('coupled_lab.html').read_text(encoding='utf-8')
    save('demo.html',template.replace('__COUPLED_DATA__',json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return report
