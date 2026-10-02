"""来源：本项目原创。贴图位置、遮挡、背面和深度保持。"""
from dataclasses import asdict
import unittest

from drone_nav.textured_card import CardTexture,render_card
from drone_nav.raycast import World,Surface,Box
from drone_nav.range_observation import render_range
from drone_nav.pinhole import Pose,Intrinsics


class TexturedCardTests(unittest.TestCase):
    def fixture(self,occluder=False):
        surfaces=[Surface('ground',0,(-10,20,-10,20)),Box('card',3,(2.,-1.,1.),(2.02,1.,3.))]
        if occluder: surfaces.append(Box('occluder',3,(1.,-1.,1.),(1.1,1.,3.)))
        return World(tuple(surfaces)),Pose((0.,0.,2.),(0.,-1.,0.),(0.,0.,-1.),(1.,0.,0.)),Intrinsics(3,3,2.,2.,1.,1.)

    def test_texture_changes_only_rgb_and_preserves_measured_geometry(self):
        world,pose,k=self.fixture(); tex=CardTexture(1,1,bytes((12,34,56)))
        original=render_range(world,pose,k)
        frame,audit=render_card(world,pose,k,card_name='card',texture=tex)
        self.assertGreater(len(audit['painted_pixels']),0)
        for key in ('depth_z_m','outcomes','pose','intrinsics','max_range_m'):
            self.assertEqual(getattr(frame,key),getattr(original,key))
        self.assertEqual(frame.rgb[4],(12,34,56))
        self.assertFalse(audit['source_photo_depth_used'])

    def test_uv_orientation_matches_camera_right_and_down(self):
        world,pose,k=self.fixture()
        tex=CardTexture(2,2,bytes((255,0,0, 0,255,0, 0,0,255, 255,255,0)))
        frame,_=render_card(world,pose,k,card_name='card',texture=tex)
        self.assertEqual([frame.rgb[i] for i in (0,2,6,8)],[(255,0,0),(0,255,0),(0,0,255),(255,255,0)])

    def test_nearer_geometry_occludes_texture(self):
        world,pose,k=self.fixture(True)
        frame,audit=render_card(world,pose,k,card_name='card',texture=CardTexture(1,1,b'abc'))
        self.assertEqual(audit['painted_pixels'],[])
        self.assertEqual(frame,render_range(world,pose,k))

    def test_back_face_keeps_original_material(self):
        world,_,k=self.fixture()
        pose=Pose.look_at((4.,0.,2.),(2.,0.,2.))
        frame,audit=render_card(world,pose,k,card_name='card',texture=CardTexture(1,1,b'abc'))
        self.assertEqual(audit['painted_pixels'],[])
        self.assertEqual(frame,render_range(world,pose,k))

    def test_sensor_failure_does_not_turn_missing_depth_into_free_space(self):
        world,pose,k=self.fixture()
        frame,_=render_card(world,pose,k,card_name='card',texture=CardTexture(1,1,b'abc'),invalid=True)
        self.assertEqual(set(frame.outcomes),{'INVALID'})
        self.assertEqual(set(frame.depth_z_m),{None})

    def test_invalid_texture_and_missing_card_fail(self):
        for args in ((0,1,b''),(1,1,b'a'),(True,1,b'abc')):
            with self.assertRaises(ValueError): CardTexture(*args)
        world,pose,k=self.fixture()
        with self.assertRaises(ValueError): render_card(world,pose,k,card_name='absent',texture=CardTexture(1,1,b'abc'))


if __name__=='__main__': unittest.main()
