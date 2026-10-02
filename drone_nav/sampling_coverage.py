"""来源：本项目原创。受控相机采样评价；不是通用视觉检测器。"""
from itertools import product
from math import isfinite

from .pinhole import Intrinsics, Pose
from .raycast import Box, World


def coverage_scenes():
    """世界目标固定后跨分辨率/视角共享，不能逐相机重置有利相位。"""
    scenes=[]
    for distance,size,pu,pv in product((3.,9.),(.005,.02,.1),(0.,.25,.5,.75),(0.,.25,.5,.75)):
        y=8-pu*distance/50;z=4-pv*distance/50
        scenes.append(dict(key=f'd{distance:g}-s{size:g}-u{pu:g}-v{pv:g}',distance_m=distance,
            size_m=size,phase=[pu,pv],low=[distance-.01,y-size/2,z-size/2],
            high=[distance+.01,y+size/2,z+size/2]))
    return scenes


def scene_world(scene):
    return World((Box('target',4,tuple(scene['low']),tuple(scene['high'])),
                  Box('background',0,(16,-50,-50),(16.1,50,50))))


def camera(width,view):
    if width not in (48,96,192) or type(width) is not int or type(view) is not int or view not in (0,1,2):
        raise ValueError('unknown study camera')
    height=width*3//4;f=50*width/96
    k=Intrinsics(width,height,f,f,width/2-.5,height/2-.5)
    y,z=((8,4),(8.12,4),(8,4.12))[view]
    return k,Pose.look_at((0,y,z),(1,y,z))


def depth_candidates(depths,*,nearer_than_m=15.):
    """本夹具仅包含 16 米背景与近处目标；仅筛选近深度，无语义身份推断。"""
    if type(nearer_than_m) not in (int,float) or not isfinite(nearer_than_m) or nearer_than_m<=0:
        raise ValueError('invalid depth threshold')
    result=[]
    for i,z in enumerate(depths):
        if z is None:continue
        if type(z) not in (int,float) or not isfinite(z) or z<=0:raise ValueError('invalid depth')
        if z<nearer_than_m:result.append(i)
    return result


def front_projection(scene,k,pose):
    """仅供独立几何评价；此场景为正对相机的轴对齐前表面。"""
    corners=[pose.project((scene['low'][0],y,z),k)
             for y,z in product((scene['low'][1],scene['high'][1]),(scene['low'][2],scene['high'][2]))]
    if any(v is None for v in corners):raise ValueError('front face behind camera')
    u0,u1=min(v[0] for v in corners),max(v[0] for v in corners)
    v0,v1=min(v[1] for v in corners),max(v[1] for v in corners)
    inside=0<=u0<=u1<=k.width-1 and 0<=v0<=v1<=k.height-1
    return dict(box=[u0,v0,u1,v1],width_px=u1-u0,height_px=v1-v0,
                front_face_sampling_condition=inside and u1-u0>=1 and v1-v0>=1)


def summarize(rows):
    """只有三个不同视角的完整同场景记录才能计入多视角对照。"""
    grouped={}
    for r in rows:
        key=(r['scene_key'],r['width'])
        if type(r['view']) is not int or r['view'] not in (0,1,2):raise ValueError('invalid view')
        views=grouped.setdefault(key,{})
        if r['view'] in views:raise ValueError('duplicate view')
        views[r['view']]=r
    result={}
    for views in grouped.values():
        if set(views)!={0,1,2}:raise ValueError('incomplete observation group')
        first=views[0]
        if any((r['distance_m'],r['size_m'],r['width'])!=(first['distance_m'],first['size_m'],first['width']) for r in views.values()):
            raise ValueError('mixed scene metadata')
        key=(first['width'],first['distance_m'],first['size_m'])
        stats=result.setdefault(key,dict(width=key[0],distance_m=key[1],size_m=key[2],positions=0,
            single_hits=0,three_view_hits=0,single_target_pixels=0,three_view_target_pixels=0,
            single_sample_count=0,three_view_sample_count=0))
        stats['positions']+=1
        stats['single_hits']+=first['target_pixels']>0
        stats['three_view_hits']+=any(r['target_pixels']>0 for r in views.values())
        stats['single_target_pixels']+=first['target_pixels']
        stats['three_view_target_pixels']+=sum(r['target_pixels'] for r in views.values())
        stats['single_sample_count']+=first['width']*first['height']
        stats['three_view_sample_count']+=sum(r['width']*r['height'] for r in views.values())
    return [dict(v,single_misses=v['positions']-v['single_hits'],three_view_misses=v['positions']-v['three_view_hits'])
            for _,v in sorted(result.items())]
