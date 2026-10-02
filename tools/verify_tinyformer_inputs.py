"""来源：本项目原创。独立抽样核对存档输入的缩放、通道、归一化和布局，不重新推理。"""
import argparse
from array import array
import gzip
import json
from pathlib import Path
import struct
import sys
from PIL import Image


def verify(out):
    request=json.loads((out/'request.json').read_text(encoding='utf-8'))
    checked=tensors=0
    positions=(0,159,319,479,639)
    for c in request['cases']:
        with Image.open(out/c['key']/'input.png') as image:
            for grid in ('1','4'):
                for t in c['plans'][grid]:
                    crop=image.convert('RGB').crop((t['x0'],t['y0'],t['x0']+t['width'],t['y0']+t['height']))
                    reference=crop.resize((640,640),Image.Resampling.BILINEAR)
                    values=array('f');values.frombytes(gzip.decompress((out/c['key']/f'grid{grid}'/f'input-{t["index"]}.gz').read_bytes()))
                    if sys.byteorder!='little':values.byteswap()
                    if len(values)!=3*640*640:raise ValueError('tensor length differs')
                    for y in positions:
                        for x in positions:
                            pixel=reference.getpixel((x,y))
                            for channel,(mean,std) in enumerate(zip((.485,.456,.406),(.229,.224,.225))):
                                # 不调用生产 prepare/f32/查表；独立写出 float32 三次运算。
                                single=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
                                expected=single(single(single(pixel[channel]/255.0)-single(mean))/single(std))
                                if values[channel*409600+y*640+x]!=expected:
                                    raise ValueError(f'input sample differs: {c["key"]} {grid} {t["index"]} {x} {y} {channel}')
                                checked+=1
                    tensors+=1
    return dict(tensors=tensors,sampled_values=checked,passed=True,
                scope='25 pixel positions per tensor, all RGB channels; shared Pillow resize; not full tensor or network equivalence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.output),ensure_ascii=False,indent=2))
