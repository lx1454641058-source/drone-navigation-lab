"""来源：本项目原创。保存图像重做决策并实际执行物理控制，再独立重放四桨指令。"""

import argparse,copy,gzip,hashlib,json
from pathlib import Path

from .classifier import ColorModel
from .native_physics import DEFAULT_DLL,QuadrotorPhysics
from .physical_navigation import navigate_physical
from .physical_vehicle import FlightBudget,PhysicalVehicle,rotate_vector,world_xml
from .pinhole import Pose
from .raycast import Box,Surface,World
from .verify_exploration import RecordedCamera


class RecordedPhysicalVehicle(PhysicalVehicle):
    def __init__(self,world,start,frames,**kwargs):
        super().__init__(world,start,**kwargs)
        self.saved_frames=frames;self.frame_index=0

    def capture(self,intrinsics,tick,aim=None):
        if self.frame_index>=len(self.saved_frames):raise ValueError('recorded physical images exhausted')
        item=self.saved_frames[self.frame_index];state=self.state()
        if state!=item['capture_state']:raise ValueError('physical camera state differs from saved observation')
        xyz=tuple(state['position']);q=state['quaternion']
        base=Pose.look_at(xyz,(xyz[0],xyz[1],0)) if aim is None else aim(xyz)
        pose=Pose(xyz,rotate_vector(base.right,q),rotate_vector(base.down,q),rotate_vector(base.forward,q))
        frame=RecordedCamera([item['frame']]).capture(pose,intrinsics,tick)
        self.frame_index+=1
        return frame


def normalize(result):
    item=copy.deepcopy(result);item['landing_images']=None
    for t in item['trace']:t['vision']=None
    return json.loads(json.dumps(item,allow_nan=False))


def saved_world(data):
    return World(tuple(Box(s['name'],s['label'],tuple(s['low']),tuple(s['high'])) if 'low' in s else
                       Surface(s['name'],s['label'],tuple(s['bounds']),s['z0'],s['slope_x'],s['slope_y'])
                       for s in data['surfaces']),data['width_m'],data['height_m'])


def verify(output,dll=DEFAULT_DLL):
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    entries={e['path']:e['sha256'] for e in manifest}
    if len(entries)!=len(manifest):raise ValueError('duplicate manifest entry')
    for name,digest in entries.items():
        path=(output/name).resolve()
        if not path.is_relative_to(output.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('invalid artifact path/digest')
    report=json.loads((output/'coupled_report.json').read_text(encoding='utf-8'))
    if report['version']!='0.10.0' or report['mode']!='coupled' or {r['key'] for r in report['results']}!={'open','building','water','dropout','ignored','wall'}:
        raise ValueError('unexpected coupled suite/version')
    required={'model.json','coupled_report.json','demo.html',*(r['key']+'-raw.json.gz' for r in report['results']),
              *(r['key']+'-world.xml' for r in report['results'])}
    if set(entries)!=required or len(report['results'])!=len({r['key'] for r in report['results']}):
        raise ValueError('incomplete manifest or duplicate scenarios')
    model=ColorModel.from_dict(json.loads((output/'model.json').read_text(encoding='utf-8')))
    checked=[]
    for saved in report['results']:
        raw=json.loads(gzip.decompress((output/(saved['key']+'-raw.json.gz')).read_bytes()))
        world=saved_world(saved['world']);start=tuple(saved['start'])
        xml=world_xml(world)
        if xml!=(output/(saved['key']+'-world.xml')).read_bytes():raise ValueError('physics world differs')
        vehicle=RecordedPhysicalVehicle(world,start,raw['frames'],dll=dll,
            dropout_tick=saved['dropout_tick'],ignore_move=saved['ignore_move'],budget=FlightBudget(**saved['budget']))
        try:
            if (vehicle.physics.version!=saved['engine_version'] or vehicle.physics.dll_sha256!=saved['dll_sha256']
                    or vehicle.physics.model_sha256!=saved['physics_model_sha256']):
                raise ValueError('physics engine/model metadata differs')
            replay=navigate_physical(vehicle,model,start,tuple(saved['goal']),max_ticks=saved['max_ticks'],previews=False)
            if normalize(replay)!=normalize({k:saved[k] for k in replay}):raise ValueError('navigation replay differs: '+saved['key'])
            if vehicle.commands!=raw['commands'] or vehicle.frame_index!=len(raw['frames']) or vehicle.audit!=saved['audit']:
                raise ValueError('control/frame/audit replay differs')
        finally:vehicle.close()
        with QuadrotorPhysics(dll,xml) as q:
            q.reset((start[0]+.5,start[1]+.5,3.5));history=[q.state()]
            for step,motors in enumerate(raw['commands'],1):
                q.step(motors)
                if step%10==0 or step==len(raw['commands']):history.append(q.state())
            if history!=saved['physics_history'] or q.state()!=saved['actual_final']:raise ValueError('raw thrust/state replay differs')
        if len(raw['commands'])!=saved['physics_steps'] or len(raw['frames'])!=saved['camera_frames']:
            raise ValueError('saved counts differ')
        checked.append(dict(key=saved['key'],frames=len(raw['frames']),steps=len(raw['commands']),terminal=saved['terminal_state']))
    return dict(verified=True,results=checked,scope='同平台图像/物理重放，不代表自然图像或实际飞行可靠性')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='重放视觉导航与物理控制联调记录')
    parser.add_argument('output',type=Path);parser.add_argument('--dll',type=Path,default=DEFAULT_DLL)
    args=parser.parse_args();print(json.dumps(verify(args.output,args.dll),ensure_ascii=False))
