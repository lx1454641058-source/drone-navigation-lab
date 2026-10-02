"""来源：本项目原创。固定 VisDrone 镜像，只取协议预选的十二张 JPG/TXT。"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from drone_nav.realvision import dump, sha
from drone_nav.object_metrics import parse_visdrone
from tools.setup_vision import put

ARCHIVE_SHA = 'abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9'
DEFAULT = Path('D:/DroneNavTools/visdrone-eval')


def select_names(names):
    images = sorted(n for n in names if n.startswith('VisDrone2019-DET-val/images/') and n.endswith('.jpg'))
    if len(images) != 548 or len(set(images)) != 548:
        raise ValueError('expected 548 unique validation images')
    groups = {}
    for name in images:
        groups.setdefault(Path(name).stem.split('_')[0], name)
    selected = sorted(groups.values(), key=lambda n: hashlib.sha256(('drone-nav-object-dev-v1:' + Path(n).name).encode()).hexdigest())[:12]
    if len(selected) != 12:
        raise ValueError('not enough filename groups')
    return selected, len(groups)


def setup(root):
    from PIL import Image, __version__ as pillow_version
    archive = root / 'mirrored-VisDrone2019-DET-val.zip'
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError('archive digest differs')
    output = root / 'selected-v1'
    if output.exists():
        raise FileExistsError('refuse to overwrite ' + str(output))
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        selected, groups = select_names(names)
        output.mkdir()
        # 在读取任何图像、标签之前独占保存固定名单。
        dump(output / 'selection.json', dict(rule='drone-nav-object-dev-v1', groups=groups, selected=selected,
                                           archive_sha256=ARCHIVE_SHA, inventory=names))
        cases = []
        for name in selected:
            key = Path(name).stem
            annotation = 'VisDrone2019-DET-val/annotations/' + key + '.txt'
            raw = z.read(name); labels = z.read(annotation)
            put(output / key / 'original.jpg', raw)
            put(output / key / 'annotations.txt', labels)
            with Image.open(io.BytesIO(raw)) as image:
                rgb = image.convert('RGB'); width, height = rgb.size
                png = io.BytesIO(); rgb.save(png, format='PNG')
                # PNG 编码可逆，但 JPEG 解码行为仍由固定 Pillow 版本决定。
                with Image.open(io.BytesIO(png.getvalue())) as check:
                    if check.tobytes() != rgb.tobytes():
                        raise ValueError('PNG conversion changed pixels')
            put(output / key / 'input.png', png.getvalue())
            records = parse_visdrone(labels.decode('utf-8-sig'), width, height)
            cases.append(dict(key=key, width=width, height=height, annotations=len(records)))
        put(output / 'protocol.md', (PROJECT / 'docs/OBJECT_EVAL_PROTOCOL.md').read_bytes())
        dump(output / 'data.json', dict(cases=cases, pillow_version=pillow_version, archive_sha256=ARCHIVE_SHA,
                                      mirror='https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip',
                                      license='CC BY-NC-SA 3.0', author='AISKYEYE team, Tianjin University'))
        dump(output / 'manifest.json', {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()})
    return dict(selected=len(cases), filename_groups=groups, labels=sum(c['annotations'] for c in cases))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='VisDrone 固定十二图开发子集准备，已有目录拒绝覆盖')
    p.add_argument('--root', type=Path, default=DEFAULT)
    print(json.dumps(setup(p.parse_args().root), ensure_ascii=False))
