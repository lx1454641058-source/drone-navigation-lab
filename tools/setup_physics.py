"""来源：本项目原创。下载固定版本官方物理引擎到独立目录，核对摘要后仅解包核心。"""

import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

VERSION = '3.3.7'
ARCHIVE = 'mujoco-3.3.7-windows-x86_64.zip'
URL = 'https://github.com/google-deepmind/mujoco/releases/download/3.3.7/'+ARCHIVE
SHA256 = '1108ad01f6a1e63d42725188404c9af50b81902c368ef026d20d5148c115a08b'
LICENSE_URL = 'https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/LICENSE'
LICENSE_SHA256 = 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'


def setup(root):
    cache = root/'work'
    cache.mkdir(parents=True,exist_ok=True)
    archive = cache/ARCHIVE
    if not archive.exists():
        # 新文件独占创建；下载中断的文件保留，摘要不符时不运行、不自动覆盖。
        with urllib.request.urlopen(URL,timeout=60) as response,archive.open('xb') as stream:
            while chunk:=response.read(1024*1024): stream.write(chunk)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('archive digest mismatch; inspect the preserved download')
    license_file = cache/'mujoco-3.3.7-LICENSE'
    if not license_file.exists():
        with urllib.request.urlopen(LICENSE_URL,timeout=60) as response,license_file.open('xb') as stream:
            stream.write(response.read())
    license_blob = license_file.read_bytes()
    if hashlib.sha256(license_blob).hexdigest()!=LICENSE_SHA256:
        raise ValueError('license digest mismatch')
    destination = root/('mujoco-'+VERSION+'-runtime')
    wanted = {'bin/mujoco.dll','THIRD_PARTY_NOTICES.txt'}
    with zipfile.ZipFile(archive) as package:
        selected = [i for i in package.infolist() if not i.is_dir() and (
            i.filename in wanted or (i.filename.startswith('include/mujoco/') and '/' not in i.filename[15:]))]
        if not wanted.issubset({i.filename for i in selected}):
            raise ValueError('core library or license missing')
        # 已存在时只验证，绝不覆盖用户或旧版本文件。
        if destination.exists():
            if (destination/'LICENSE').read_bytes()!=license_blob:
                raise ValueError('existing license differs')
            for item in selected:
                if (destination/item.filename).read_bytes()!=package.read(item):
                    raise ValueError('existing runtime differs; no files overwritten')
        else:
            destination.mkdir(parents=True,exist_ok=False)
            (destination/'LICENSE').write_bytes(license_blob)
            for item in selected:
                target = destination/item.filename
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError('archive path escapes runtime')
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream: stream.write(package.read(item))
    dll = destination/'bin/mujoco.dll'
    print('runtime:',destination.resolve())
    print('dll_sha256:',hashlib.sha256(dll.read_bytes()).hexdigest())


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='准备经核验的 MuJoCo 3.3.7 Windows 核心，不修改系统配置')
    parser.add_argument('--root',type=Path,default=Path('D:/DroneNavTools'))
    setup(parser.parse_args().root)
