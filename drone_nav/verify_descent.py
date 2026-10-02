"""来源：本项目原创。保存图像/间距读数重做决策，独立重放推力并核对全部状态。"""

import argparse,copy,gzip,hashlib,json
from pathlib import Path

from .classifier import ColorModel
from .descent_vehicle import DescentVehicle
from .native_physics import DEFAULT_DLL,QuadrotorPhysics
from .physical_navigation import navigate_physical
from .physical_vehicle import world_xml
from .verify_coupled import RecordedPhysicalVehicle,normalize,saved_world
from .visual_descent import DescentConfig,descend_after_navigation


class RecordedDescentVehicle(DescentVehicle):
    def __init__(self,world,start,raw,**kwargs):
        super().__init__(world,start,**kwargs)
        self.saved_frames=raw['frames'];self.frame_index=0
        self.saved_contacts=raw['contact_readings'];self.contact_index=0

    capture=RecordedPhysicalVehicle.capture

    def contact_gap(self):
        if self.contact_index>=len(self.saved_contacts):raise ValueError('saved contact readings exhausted')
        saved=self.saved_contacts[self.contact_index]
        # 保留同样的 mj_forward 调用时序，再比较探针；不把保存值当物理真值写回。
        value=super().contact_gap()
        if self.contact_readings[-1]!=saved:raise ValueError('contact evidence differs')
        self.contact_index+=1
        return value


def normalize_descent(result):
    result=copy.deepcopy(result)
    for e in result['events']:e['images']=None
    return json.loads(json.dumps(result,allow_nan=False))


def replay_raw(q,raw,start):
    """几何探针包含 forward；按保存的步号复现调用顺序而不是忽略其副作用。"""
    q.reset((start[0]+.5,start[1]+.5,3.5));history=[q.state()];index=0
    readings=raw['contact_readings']
    for step,motors in enumerate(raw['commands'],1):
        q.step(motors)
        if step%10==0 or step==len(raw['commands']):history.append(q.state())
        while index<len(readings) and readings[index]['step']==step:
            if q.ground_distance()!=readings[index]['truth_gap_m'] or q.state()['time_s']!=readings[index]['time_s']:
                raise ValueError('raw geometric probe differs')
            index+=1
    if index!=len(readings):raise ValueError('unconsumed geometric readings')
    return history


def verify(output,dll=DEFAULT_DLL):
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    entries={e['path']:e['sha256'] for e in manifest}
    if len(entries)!=len(manifest):raise ValueError('duplicate manifest')
    for name,digest in entries.items():
        path=(output/name).resolve()
        if not path.is_relative_to(output.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('artifact path/digest differs')
    report=json.loads((output/'descent_report.json').read_text(encoding='utf-8'))
    keys={'paved','building','water','bump','depth','semantic','contact','postcontact','expired'}
    if report['mode']!='descent' or report['version']!='0.11.0' or len(report['results'])!=9 or {r['key'] for r in report['results']}!=keys:
        raise ValueError('unexpected descent suite')
    if set(entries)!={'model.json','demo.html','descent_report.json',*(k+'-raw.json.gz' for k in keys),*(k+'-world.xml' for k in keys)}:
        raise ValueError('incomplete descent manifest')
    model=ColorModel.from_dict(json.loads((output/'model.json').read_text(encoding='utf-8')));checked=[]
    for saved in report['results']:
        raw=json.loads(gzip.decompress((output/(saved['key']+'-raw.json.gz')).read_bytes()))
        world=saved_world(saved['world']);start=tuple(saved['start']);xml=world_xml(world)
        if xml!=(output/(saved['key']+'-world.xml')).read_bytes():raise ValueError('world XML differs')
        vehicle=RecordedDescentVehicle(world,start,raw,dll=dll,fault=saved['fault'],fault_after_s=saved['fault_after_s'])
        try:
            if (vehicle.physics.version!=saved['engine_version'] or vehicle.physics.dll_sha256!=saved['dll_sha256']
                    or vehicle.physics.model_sha256!=saved['physics_model_sha256']):raise ValueError('engine/model differs')
            nav=navigate_physical(vehicle,model,start,tuple(saved['goal']),max_ticks=saved['navigation']['max_ticks'],previews=False)
            descent=descend_after_navigation(vehicle,model,nav,config=DescentConfig(**saved['descent']['config']),previews=False)
            if normalize(nav)!=normalize(saved['navigation']) or normalize_descent(descent)!=normalize_descent(saved['descent']):
                raise ValueError('decision replay differs: '+saved['key'])
            if (vehicle.commands!=raw['commands'] or vehicle.frame_index!=len(raw['frames'])
                    or vehicle.contact_index!=len(raw['contact_readings']) or vehicle.audit!=saved['audit']):
                raise ValueError('replay command/frame/contact/audit differs')
        finally:vehicle.close()
        with QuadrotorPhysics(dll,xml) as q:
            history=replay_raw(q,raw,start)
            if history!=saved['history'] or q.state()!=saved['actual_final']:raise ValueError('raw state replay differs')
        if len(raw['commands'])!=saved['steps'] or len(raw['frames'])!=saved['camera_frames']:raise ValueError('count mismatch')
        checked.append(dict(key=saved['key'],status=descent['status'],frames=len(raw['frames']),steps=len(raw['commands'])))
    return dict(verified=True,results=checked,scope='固定引擎同平台重放，非真实配送证明')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='重放视觉下降实验')
    parser.add_argument('output',type=Path);parser.add_argument('--dll',type=Path,default=DEFAULT_DLL)
    args=parser.parse_args();print(json.dumps(verify(args.output,args.dll),ensure_ascii=False))
