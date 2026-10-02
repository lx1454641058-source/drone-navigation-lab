"""来源：本项目原创。将已许可照片映射到仿真薄板前表面；不改射线深度。"""
from dataclasses import dataclass
from math import floor

from .pinhole import PerspectiveFrame
from .raycast import Box, render
from .range_observation import RangeFrame


@dataclass(frozen=True)
class CardTexture:
    width: int
    height: int
    rgb_bytes: bytes

    def __post_init__(self):
        if (type(self.width) is not int or type(self.height) is not int or min(self.width,self.height)<1
            or self.width*self.height>1_000_000 or type(self.rgb_bytes) is not bytes
            or len(self.rgb_bytes)!=self.width*self.height*3):
            raise ValueError('bounded immutable RGB texture required')


def render_card(world, pose, intrinsics, *, card_name, texture, tick=0, seed=1701,
                max_range_m=30., invalid=False):
    """Renderer-only access to world truth; consumers receive ordinary RGB-D.

    Only rays whose nearest hit is the card's low-X face receive its texture.
    Occluders, backs, sides, depth, and radial range retain the old raycast result.
    Texture depicts an appearance on a plane, not the source photo's 3-D scene.
    """
    cards=[surface for surface in world.surfaces if isinstance(surface,Box) and surface.name==card_name]
    if len(cards)!=1 or type(texture) is not CardTexture or type(invalid) is not bool:
        raise ValueError('one physical card and immutable texture required')
    card=cards[0]
    frame,truth=render(world,pose,intrinsics,tick=tick,seed=seed,max_range_m=max_range_m)
    rgb=list(frame.rgb)
    painted=[]
    for index,(name,point) in enumerate(zip(truth.objects,truth.points)):
        if name!=card_name or abs(point[0]-card.low[0])>1e-9:
            continue
        u=(card.high[1]-point[1])/(card.high[1]-card.low[1])
        v=(card.high[2]-point[2])/(card.high[2]-card.low[2])
        tx=min(texture.width-1,max(0,floor(u*texture.width)))
        ty=min(texture.height-1,max(0,floor(v*texture.height)))
        offset=(ty*texture.width+tx)*3
        rgb[index]=tuple(texture.rgb_bytes[offset:offset+3])
        painted.append(index)
    outcomes=tuple('INVALID' if invalid else 'NO_HIT' if z is None else 'HIT' for z in frame.depth_z_m)
    result=RangeFrame(intrinsics,pose,tuple(rgb),(None,)*len(rgb) if invalid else frame.depth_z_m,
                      tick,outcomes,max_range_m)
    result.validate()
    return result,dict(painted_pixels=painted,card_name=card_name,texture_surface='low-X face',
        source_photo_depth_used=False,physical_scene_is_a_textured_plane=True)
