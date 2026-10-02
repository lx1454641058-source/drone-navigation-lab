"""来源：本项目原创。巡航、视觉下降与失败中断的完整开发实验。"""

import gzip,hashlib,json
from dataclasses import asdict
from pathlib import Path

from .descent_vehicle import DescentVehicle
from .native_physics import DEFAULT_DLL
from .physical_navigation import navigate_physical
from .physical_vehicle import world_xml
from .raycast import Box,Surface,World
from .visual_descent import DescentConfig,descend_after_navigation
from .vision_experiment import train_model


def descent_cases():
    ground=Surface('ground',0,(-40,60,-40,56));base=World((ground,))
    return [dict(key='paved',title='铺装地面：到达、下降、接地',world=base),
            dict(key='building',title='绕过建筑后下降接地',world=World((ground,Box('building',3,(8,6,0),(10,10,6)))),goal=(16,8)),
            dict(key='water',title='水面：巡航末端即拒降',world=World((Surface('water',2,ground.bounds),))),
            dict(key='bump',title='同色小凸起：通道复查拒绝',world=World((ground,Box('bump',0,(4.3,8.3,0),(4.7,8.7,.10))))),
            dict(key='depth',title='下降途中深度失效',world=base,fault='depth'),
            dict(key='semantic',title='下降途中颜色异常注入',world=base,fault='semantic'),
            dict(key='contact',title='接地前间距探针失效',world=base,fault='contact',fault_after_s=26),
            dict(key='postcontact',title='停桨后间距探针失效',world=base,fault='contact',fault_after_s=27.5),
            dict(key='expired',title='局部视野下旧通道证据过期',world=base,config=DescentConfig(certificate_ttl_s=1.0))]


def run_case(case,model,*,dll=DEFAULT_DLL,previews=True):
    start=(3,8);goal=case.get('goal',(4,8));config=case.get('config',DescentConfig())
    vehicle=DescentVehicle(case['world'],start,dll=dll,fault=case.get('fault'),fault_after_s=case.get('fault_after_s',4.0))
    try:
        nav=navigate_physical(vehicle,model,start,goal,previews=previews)
        descent=descend_after_navigation(vehicle,model,nav,config=config,previews=previews)
        if vehicle.history[-1]!=vehicle.state():vehicle.history.append(vehicle.state())
        report=dict(key=case['key'],title=case['title'],world=asdict(case['world']),start=start,goal=goal,
                    fault=case.get('fault'),fault_after_s=case.get('fault_after_s',4.0),
                    navigation=nav,descent=descent,engine_version=vehicle.physics.version,
                    dll_sha256=vehicle.physics.dll_sha256,physics_model_sha256=vehicle.physics.model_sha256,
                    history=vehicle.history,actions=vehicle.actions,audit=vehicle.audit,
                    steps=vehicle.steps,camera_frames=len(vehicle.frames),actual_final=vehicle.state(),
                    metrics=dict(min_ground_gap_m=min((r['truth_gap_m'] for r in vehicle.contact_readings),default=None),
                                 descent_s=descent['ended_at_s']-descent['started_at_s'],
                                 observations=len(descent['events']),
                                 local_only_observations=sum(not e['full_corridor'] for e in descent['events'])))
        raw=dict(frames=vehicle.frames,commands=vehicle.commands,contact_readings=vehicle.contact_readings)
        return report,raw
    finally:vehicle.close()


def run_suite(output,dll=DEFAULT_DLL):
    output.mkdir(parents=True,exist_ok=False);model=train_model();manifest=[];results=[]
    def save(name,data):
        (output/name).write_bytes(data);manifest.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest()))
    save('model.json',json.dumps(model.to_dict()).encode('utf-8'))
    for case in descent_cases():
        report,raw=run_case(case,model,dll=dll)
        save(case['key']+'-raw.json.gz',gzip.compress(json.dumps(raw,allow_nan=False,separators=(',',':')).encode('utf-8'),mtime=0))
        save(case['key']+'-world.xml',world_xml(case['world']));results.append(report)
        print(case['key'],report['navigation']['terminal_state'],report['descent']['status'],report['descent']['reason'],report['metrics'],flush=True)
    report=dict(version='0.11.0',mode='descent',results=results,
                scope='静态原创场景、合成视觉和精确仿真状态；从空中开始，接地不代表完成放餐或实际配送。')
    save('descent_report.json',json.dumps(report,ensure_ascii=False,allow_nan=False).encode('utf-8'))
    template=Path(__file__).with_name('descent_lab.html').read_text(encoding='utf-8')
    save('demo.html',template.replace('__DESCENT_DATA__',json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).encode('utf-8'))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return report
