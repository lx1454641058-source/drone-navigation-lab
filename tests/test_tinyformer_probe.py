"""来源：本项目原创。独立的颜色/坐标期望和错误拒绝；不调用网络模型。"""
from array import array
from pathlib import Path
import subprocess
import unittest
from PIL import Image
from tools.tinyformer_probe import prepare


class TinyFormerTests(unittest.TestCase):
    def test_red_channels_and_layout(self):
        im=Image.new('RGB',(2,3),(255,0,0))
        data=array('f');data.frombytes(prepare(im,dict(x0=0,y0=0,width=2,height=3)))
        self.assertEqual(len(data),3*640*640)
        # 手算恒定红色，不依赖生产查表函数计算期望。
        for c,value in enumerate((2.2489083,-2.0357143,-1.8044444)):
            for offset in (0,639,640*640-1):self.assertAlmostEqual(data[c*640*640+offset],value,places=5)

    def test_crop_before_resize(self):
        im=Image.new('RGB',(4,2),(255,0,0))
        im.paste((0,255,0),(2,0,4,2))
        a=prepare(im,dict(x0=2,y0=0,width=2,height=2))
        b=prepare(Image.new('RGB',(2,2),(0,255,0)),dict(x0=0,y0=0,width=2,height=2))
        self.assertEqual(a,b)

    def test_invalid_windows(self):
        im=Image.new('RGB',(4,2))
        for w in [dict(x0=-1,y0=0,width=2,height=2),dict(x0=3,y0=0,width=2,height=2),
                  dict(x0=0,y0=0,width=0,height=2),dict(x0=0.5,y0=0,width=2,height=2)]:
            with self.assertRaises(ValueError):prepare(im,w)

    def test_javascript_decoder(self):
        result=subprocess.run(['node',str(Path(__file__).with_name('tinyformer_math.cjs'))],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
