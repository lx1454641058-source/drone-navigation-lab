"""来源：本项目原创。已有 Pillow 仅用于独立输入对照，不成为应用依赖。"""
import argparse
from array import array
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from drone_nav.realvision import asset_root, validate_assets, sha, dump


def difference(a,b):
    if len(a)!=len(b) or not a:
        raise ValueError('empty or mismatched arrays')
    return dict(count=len(a),different=sum(x!=y for x,y in zip(a,b)),
                max_abs=max(abs(x-y) for x,y in zip(a,b)))


def floats(data):
    a=array('f');a.frombytes(data)
    if sys.byteorder!='little':a.byteswap()
    return a


def run(out):
    if out.exists():raise FileExistsError('refuse to overwrite: '+str(out))
    from PIL import Image, __version__ as pillow_version
    root=asset_root().resolve();validate_assets(root)
    city=Path('D:/DroneNavTools/vision-cityscapes-probe')
    if sha(city/'model.onnx')!='4dc3ab8ed0c6d724ad58fa7ddfeea0f38c85f2e5c0aa544c88a83bc233f95292':
        raise ValueError('candidate model differs')
    out.mkdir(parents=True,exist_ok=False)
    decoded={}
    pre=json.loads((root/'model/preprocessor_config.json').read_text(encoding='utf-8'))
    for seq in (1,2):
        key=f'seq{seq}_000000';d=out/key;d.mkdir()
        with Image.open(root/f'data/images/train/{key}.png') as raw:
            im=raw.convert('RGB')
        with Image.open(root/f'data/masks/train/{key}.png') as raw:
            label=raw.convert('RGBA')
        if im.size!=(3840,2160) or label.size!=im.size:raise ValueError('unexpected development shape')
        decoded[key]=dict(image_sha=hashlib.sha256(im.convert('RGBA').tobytes()).hexdigest(),
                          label_sha=hashlib.sha256(label.tobytes()).hexdigest(),width=im.width,height=im.height)
        rgb=im.resize((512,512),Image.Resampling.BILINEAR).tobytes()
        (d/'pillow.rgb').write_bytes(rgb)
        # 明确 NCHW 平面顺序；与 JS 相同公式，但图像读取、缩放和序列化独立。
        values=array('f',((rgb[i]*pre['rescale_factor']-pre['image_mean'][c])/pre['image_std'][c]
                         for c in range(3) for i in range(c,len(rgb),3)))
        if sys.byteorder!='little':values.byteswap()
        (d/'pillow.f32').write_bytes(values.tobytes())
        # 两张开发图均恰好缩小四倍，像素中心对应 4*x+2、4*y+2；不调用旧 labelGrid。
        pixels=label.load()
        truth=bytes(pixels[4*x+2,4*y+2][0] for y in range(540) for x in range(960))
        (d/'pillow.label').write_bytes(truth)
    worker=Path(__file__).with_suffix('.cjs')
    sources={str(p.relative_to(PROJECT)):sha(p) for p in [Path(__file__),worker,PROJECT/'drone_nav/segmentation_worker.cjs']}
    q=dict(root=str(root),city=str(city.resolve()),output=str(out.resolve()))
    dump(out/'request.json',q)
    subprocess.run(['node',str(worker)],input=json.dumps(q),text=True,check=True,timeout=300)
    results=[]
    for seq in (1,2):
        key=f'seq{seq}_000000';d=out/key
        if json.loads((d/'decoded.json').read_text())!=decoded[key]:raise ValueError('decoded images differ')
        if (d/'js.label').read_bytes()!=(d/'pillow.label').read_bytes():raise ValueError('label sampling differs')
        row=dict(key=key,decoded_equal=True,label_equal=True,
                 rgb=difference((d/'js.rgb').read_bytes(),(d/'pillow.rgb').read_bytes()),
                 normalized=difference(floats((d/'js.f32').read_bytes()),floats((d/'pillow.f32').read_bytes())),models={})
        for model in ('ade','city'):
            base=d/f'{model}-all-js'
            archived=PROJECT/('work/run-v13-scale-validated' if model=='ade' else 'work/cityscapes-probe-01')
            rel=key+('/whole/ade.u8' if model=='ade' else '/grid1/city.u8')
            manifest=json.loads((archived/'manifest.json').read_text())
            if sha(archived/rel)!=manifest[rel] or (archived/rel).read_bytes()!=base.with_suffix('.u8').read_bytes():
                raise ValueError('original prediction differs')
            entries={}
            for variant in ('disabled-js','disabled-pillow'):
                target=d/f'{model}-{variant}'
                entries[variant]=dict(classes=difference(base.with_suffix('.u8').read_bytes(),target.with_suffix('.u8').read_bytes()),
                    scores=difference(floats(gzip.decompress(base.with_suffix('.f32.gz').read_bytes())),
                                      floats(gzip.decompress(target.with_suffix('.f32.gz').read_bytes()))))
            row['models'][model]=entries
        results.append(row)
    dump(out/'report.json',dict(kind='development_io_audit',pillow_version=pillow_version,sources=sources,
        models={'ade':sha(root/'model/model.onnx'),'city':sha(city/'model.onnx')},results=results,
        limits=['同一 ONNX 网络和 CPU 运行库，不能证明 NVIDIA 原始网络等价',
                '仅两个开发帧；解码一致不证明镜像转换前的原始标签正确',
                '关闭运行优化及轻微缩放差异不等于修复航拍域差异']))
    dump(out/'manifest.json',{str(p.relative_to(out)).replace('\\','/'):sha(p) for p in out.rglob('*') if p.is_file()})
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='只核对两个开发帧；需要本机已核验许可的 Pillow')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.output),ensure_ascii=False,indent=2))
