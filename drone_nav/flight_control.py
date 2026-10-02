"""来源：本项目原创。理想四旋翼的位置/姿态 PD 控制与四电机推力分配。

PD 指根据目标误差和变化速度调节控制量。本模块参数仅匹配原创开发机体。
"""

from math import atan2,cos,isfinite,sin,sqrt

MASS = 1.4
GRAVITY = 9.81
ARM = .18
YAW_RATIO = .015


def quaternion_product(a,b):
    w,x,y,z = a
    v,i,j,k = b
    return (w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v)


def euler_quaternion(roll,pitch,yaw=0):
    cr,sr,cp,sp,cy,sy = cos(roll/2),sin(roll/2),cos(pitch/2),sin(pitch/2),cos(yaw/2),sin(yaw/2)
    return (cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy)


def allocate(total,torques):
    tx,ty,tz = torques
    raw = [total/4 + sy*tx/(4*ARM) - sx*ty/(4*ARM) + spin*tz/(4*YAW_RATIO)
           for sx,sy,spin in ((1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1))]
    return [min(8.0,max(0.0,v)) for v in raw],any(v<0 or v>8 for v in raw)


def control(state,target):
    if len(target)!=3 or any(type(v) not in (int,float) or not isfinite(v) for v in target):
        raise ValueError('target must be finite xyz')
    position,velocity,q,omega = (state[k] for k in ('position','velocity','quaternion','angular_velocity'))
    acceleration = [1.8*(target[i]-position[i])-2.8*velocity[i] for i in range(3)]
    ax,ay = (min(2.5,max(-2.5,v)) for v in acceleration[:2])
    az = max(2.0,min(15.0,GRAVITY+acceleration[2]))
    desired = euler_quaternion(atan2(-ay,az),atan2(ax,sqrt(ay*ay+az*az)))
    error = quaternion_product((q[0],-q[1],-q[2],-q[3]),desired)
    sign = 1 if error[0]>=0 else -1
    torque = [2*gain*sign*error[i+1]-damping*omega[i]
              for i,(gain,damping) in enumerate(((.8,.22),(.8,.22),(.4,.18)))]
    up_z = 1-2*(q[1]*q[1]+q[2]*q[2])
    total = MASS*az/max(.5,up_z)
    motors,saturated = allocate(total,torque)
    return motors,dict(saturated=saturated,target=list(target),desired_quaternion=desired,
                       requested_total_thrust_n=total,requested_torque_nm=torque)
