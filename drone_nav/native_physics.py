"""来源：本项目原创适配器；MuJoCo 3.3.7 C API 及许可见 docs/PHYSICS_ENVIRONMENT.md。

仅通过公开函数访问不透明指针，不猜测 C 结构体的内存布局，不依赖 NumPy。
"""

import ctypes as ct
import hashlib
from math import isfinite,sqrt
from pathlib import Path

DEFAULT_DLL = Path('D:/DroneNavTools/mujoco-3.3.7-runtime/bin/mujoco.dll')
DLL_SHA256 = '33803eead04041b76ec658492d79a4b1166bf59e8512f6207d7feeb01c1be278'
MODEL = Path(__file__).with_name('quadrotor.xml')
STATE = 1|2|4  # mjSTATE_TIME | mjSTATE_QPOS | mjSTATE_QVEL，固定 3.3.7 头文件。
CTRL = 1<<5


class QuadrotorPhysics:
    dt = .002

    def __init__(self,dll=DEFAULT_DLL,model_xml=None):
        path = Path(dll).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=DLL_SHA256:
            raise ValueError('MuJoCo DLL differs from the verified Windows 3.3.7 core')
        self.dll_sha256 = DLL_SHA256
        self.lib = ct.CDLL(str(path))
        p,d = ct.c_void_p,ct.POINTER(ct.c_double)
        signatures = {
            'mj_version':([],ct.c_int),'mj_versionString':([],ct.c_char_p),
            'mj_parseXMLString':([ct.c_char_p,p,ct.c_char_p,ct.c_int],p),
            'mj_compile':([p,p],p),'mjs_getError':([p],ct.c_char_p),
            'mj_deleteSpec':([p],None),'mj_makeData':([p],p),
            'mj_deleteData':([p],None),'mj_deleteModel':([p],None),
            'mj_resetData':([p,p],None),'mj_forward':([p,p],None),'mj_step':([p,p],None),
            'mj_stateSize':([p,ct.c_uint],ct.c_int),
            'mj_getState':([p,p,d,ct.c_uint],None),'mj_setState':([p,p,d,ct.c_uint],None),
            'mj_name2id':([p,ct.c_int,ct.c_char_p],ct.c_int),
            'mj_geomDistance':([p,p,ct.c_int,ct.c_int,ct.c_double,d],ct.c_double),
        }
        for name,(arguments,result) in signatures.items():
            function = getattr(self.lib,name)
            function.argtypes,function.restype = arguments,result
        if self.lib.mj_version()!=337: raise ValueError('unsupported native API version')
        self.version = self.lib.mj_versionString().decode('ascii')
        self.model = self.data = None
        xml = MODEL.read_bytes() if model_xml is None else model_xml
        self.model_sha256 = hashlib.sha256(xml).hexdigest()
        error = ct.create_string_buffer(2048)
        spec = self.lib.mj_parseXMLString(xml,None,error,len(error))
        if not spec: raise ValueError(error.value.decode('utf-8',errors='replace'))
        try:
            self.model = self.lib.mj_compile(spec,None)
            if not self.model: raise ValueError(self.lib.mjs_getError(spec).decode('utf-8',errors='replace'))
        finally:
            self.lib.mj_deleteSpec(spec)
        try:
            if self.lib.mj_stateSize(self.model,STATE)!=14 or self.lib.mj_stateSize(self.model,CTRL)!=4:
                raise ValueError('expected one free body and four actuators')
            self.data = self.lib.mj_makeData(self.model)
            if not self.data: raise MemoryError('MuJoCo data allocation failed')
            self.ground = self.lib.mj_name2id(self.model,5,b'ground')
            self.body = self.lib.mj_name2id(self.model,5,b'fuselage')
            if min(self.ground,self.body)<0: raise ValueError('required geometry missing')
            self.lib.mj_forward(self.model,self.data)
        except Exception:
            self.close()
            raise

    def _open(self):
        if not self.model or not self.data: raise RuntimeError('physics instance is closed')

    def close(self):
        if self.data: self.lib.mj_deleteData(self.data); self.data = None
        if self.model: self.lib.mj_deleteModel(self.model); self.model = None

    def __enter__(self): return self

    def __exit__(self,*args): self.close()

    def reset(self,position=(0,0,1),quaternion=(1,0,0,0)):
        self._open()
        values = (*position,*quaternion)
        if len(position)!=3 or len(quaternion)!=4 or any(type(v) not in (float,int) or not isfinite(v) for v in values):
            raise ValueError('invalid initial pose')
        norm = sqrt(sum(x*x for x in quaternion))
        if abs(norm-1)>1e-6: raise ValueError('initial quaternion must be normalized')
        self.lib.mj_resetData(self.model,self.data)
        state = (ct.c_double*14)(0,*values,*([0]*6))
        self.lib.mj_setState(self.model,self.data,state,STATE)
        self.lib.mj_forward(self.model,self.data)

    def state(self):
        self._open()
        values = (ct.c_double*14)()
        self.lib.mj_getState(self.model,self.data,values,STATE)
        if not all(isfinite(v) for v in values): raise ValueError('nonfinite physical state')
        return dict(time_s=values[0],position=list(values[1:4]),quaternion=list(values[4:8]),
                    velocity=list(values[8:11]),angular_velocity=list(values[11:14]))

    def step(self,rotor_thrusts):
        self._open()
        if len(rotor_thrusts)!=4 or any(type(v) not in (int,float) or not isfinite(v) or not 0<=v<=8 for v in rotor_thrusts):
            raise ValueError('four finite rotor thrusts within 0..8 N required')
        controls = (ct.c_double*4)(*rotor_thrusts)
        self.lib.mj_setState(self.model,self.data,controls,CTRL)
        self.lib.mj_step(self.model,self.data)

    def ground_distance(self):
        self._open()
        # step 后刷新几何到当前积分状态；返回有符号间距，不冒充接触力传感器。
        self.lib.mj_forward(self.model,self.data)
        return self.lib.mj_geomDistance(self.model,self.data,self.ground,self.body,100,None)
