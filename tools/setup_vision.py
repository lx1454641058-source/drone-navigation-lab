"""来源：本项目原创。固定版本本地视觉实验资源；核验后解包，不执行 npm 安装脚本。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

DEFAULT_ROOT = Path('D:/DroneNavTools/vision-v12')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def put(path, data):
    """已有文件只核对；新文件独占创建，防止覆盖既有实验资源。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'existing file differs: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(data)


def contained(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative:
        raise ValueError('invalid asset path')
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('asset path escapes destination')
    return target


def setup(root, reuse=None):
    lock_path = Path(__file__).with_name('vision_assets.json')
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    cached = {}
    if reuse and reuse.is_dir():
        for p in reuse.iterdir():
            if p.is_file():
                cached[digest(p.read_bytes())] = p

    def acquire(asset):
        target = contained(root, asset['path'])
        if target.exists():
            body = target.read_bytes()
        elif asset['sha256'] in cached:
            body = cached[asset['sha256']].read_bytes()
        else:
            with urllib.request.urlopen(asset['url'], timeout=90) as response:
                body = response.read(asset['bytes'] + 1)
        if len(body) != asset['bytes'] or digest(body) != asset['sha256']:
            raise ValueError(f"resource hash/size mismatch: {asset['path']}")
        put(target, body)
        print('verified', asset['path'], len(body), flush=True)

    # 仅并行下载独立资源；全部通过后才进入解包步骤。
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(acquire, lock['assets']))
    runtime = root / 'runtime' / 'node_modules'
    for name, version in [('onnxruntime-node', '1.22.0'),
                          ('onnxruntime-common', '1.22.0'), ('pngjs', '7.0.0')]:
        with tarfile.open(root / 'packages' / f'{name}-{version}.tgz') as package:
            for entry in package.getmembers():
                if not entry.isfile():
                    continue
                if not entry.name.startswith('package/'):
                    raise ValueError('unexpected package layout')
                rel = entry.name[len('package/'):]
                if name == 'onnxruntime-node':
                    wanted = (rel == 'package.json' or rel.startswith('dist/') or rel in (
                        'bin/napi-v6/win32/x64/onnxruntime.dll',
                        'bin/napi-v6/win32/x64/onnxruntime_binding.node'))
                elif name == 'onnxruntime-common':
                    wanted = rel == 'package.json' or rel.startswith('dist/cjs/')
                else:
                    wanted = rel in ('LICENSE', 'package.json') or rel.startswith('lib/')
                if wanted:
                    put(contained(runtime / name, rel), package.extractfile(entry).read())
        if name.startswith('onnxruntime-'):
            put(runtime / name / 'LICENSE', (root / 'licenses/LICENSE').read_bytes())
            put(runtime / name / 'ThirdPartyNotices.txt',
                (root / 'licenses/ThirdPartyNotices.txt').read_bytes())
    put(root / 'assets-lock.json', lock_path.read_bytes())
    files = {str(p.relative_to(root)).replace('\\', '/'): digest(p.read_bytes())
             for p in (root / 'runtime').rglob('*') if p.is_file()}
    put(root / 'runtime-lock.json', (json.dumps(files, indent=2)+'\n').encode())
    print('ready:', root.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='准备研究用途视觉资源：约 249 MB，Windows x64 CPU')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--reuse', type=Path, help='只读复用此前已下载且摘要相同的文件')
    args = parser.parse_args()
    setup(args.root, args.reuse)
