"""来源：本项目原创。只读下载官方 ROS bag 的固定时间子集，无 ROS 依赖。

格式依据 ROS 官方 rosbag / sensor_msgs 定义，未复制其实现。
保留原始分块和 Image 消息；彩色数据无损转 PNG，深度保留 float32。
"""
import argparse
import bz2
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drone_nav.realvision import dump, sha
from drone_nav.tum_rgbd import nearest, rows
from tools.tum_rgbd_experiment import OFFSETS, manifest

URL = 'https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.bag'
TOTAL = 578591964
INDEX = 578341855


class Reader:
    def __init__(self, data): self.data, self.pos = data, 0
    def take(self, n):
        if n < 0 or self.pos+n>len(self.data): raise ValueError('truncated ROS record')
        result=self.data[self.pos:self.pos+n]; self.pos+=n; return result
    def uint(self): return struct.unpack('<I',self.take(4))[0]
    def sized(self): return self.take(self.uint())


def fields(data):
    reader=Reader(data); result={}
    while reader.pos<len(data):
        key,value=reader.sized().split(b'=',1)
        if key in result: raise ValueError('duplicate ROS header field')
        result[key]=value
    return result


def records(data):
    reader=Reader(data)
    while reader.pos<len(data):
        yield fields(reader.sized()),reader.sized()


def uint(data): return struct.unpack('<I',data)[0]
def stamp(data):
    sec,nsec=struct.unpack('<II',data)
    if nsec>=1_000_000_000: raise ValueError('invalid ROS nanoseconds')
    return sec+nsec/1e9


def index_records(data):
    connections={}; chunks=[]
    for header,body in records(data):
        if header[b'op']==b'\x07':
            connections[uint(header[b'conn'])]=dict(topic=header[b'topic'].decode(),
                type=fields(body)[b'type'].decode())
        elif header[b'op']==b'\x06':
            if uint(header[b'ver'])!=1 or len(body)!=8*uint(header[b'count']):
                raise ValueError('unsupported chunk info')
            chunks.append(dict(start=stamp(header[b'start_time']),end=stamp(header[b'end_time']),
                pos=struct.unpack('<Q',header[b'chunk_pos'])[0],
                connections=dict(struct.iter_unpack('<II',body))))
        else: raise ValueError('unexpected index record')
    if len(chunks)!=1693 or len(connections)!=6: raise ValueError('fixed bag index differs')
    for i,c in enumerate(chunks):
        c['stop'] = chunks[i+1]['pos'] if i+1<len(chunks) else INDEX
        if not 0<c['stop']-c['pos']<2_000_000: raise ValueError('unexpected chunk range')
    return connections,chunks


def fetch_range(directory,start,stop):
    destination=directory/f'{start}-{stop}.bin'
    header=directory/f'{start}-{stop}.http'
    expected=stop-start
    if destination.exists():
        if destination.stat().st_size!=expected: raise ValueError('incomplete existing range; keep original')
    else:
        subprocess.run(['curl.exe','--fail','--silent','--show-error','--range',f'{start}-{stop-1}',
            '--max-filesize',str(expected),'--max-time','180','--dump-header',str(header),
            '--output',str(destination),URL],check=True,timeout=190)
    text=header.read_text(encoding='ascii').lower()
    if (f'content-range: bytes {start}-{stop-1}/{TOTAL}' not in text
            or destination.stat().st_size!=expected or '206 partial content' not in text):
        raise ValueError('HTTP range response differs')
    return destination


def images(chunk_path, connections):
    header,body=next(records(chunk_path.read_bytes()))
    if header[b'op']!=b'\x05': raise ValueError('not a bag chunk')
    if header[b'compression']==b'bz2': raw=bz2.decompress(body)
    elif header[b'compression']==b'none': raw=body
    else: raise ValueError('unsupported bag compression')
    if len(raw)!=uint(header[b'size']) or len(raw)>8_000_000: raise ValueError('invalid decompressed size')
    result=[]
    for h,d in records(raw):
        if h[b'op']!=b'\x02': continue
        connection=connections[uint(h[b'conn'])]
        if connection['type']!='sensor_msgs/Image': continue
        r=Reader(d); sequence=r.uint(); captured=stamp(r.take(8)); frame=r.sized().decode()
        height,width=r.uint(),r.uint(); encoding=r.sized().decode(); big=r.take(1)[0]; step=r.uint(); pixels=r.sized()
        if (r.pos!=len(d) or (width,height)!=(640,480) or len(pixels)!=height*step or big!=0):
            raise ValueError('unsupported image layout')
        if (encoding,step) not in [('rgb8',1920),('bgr8',1920),('32FC1',2560)]:
            raise ValueError('unsupported image encoding '+encoding)
        result.append(dict(topic=connection['topic'],captured_at_s=captured,
            bag_at_s=stamp(h[b'time']),sequence=sequence,frame_id=frame,encoding=encoding,
            width=width,height=height,step=step,raw_message=d,pixels=pixels,chunk_file=chunk_path.name))
    return result


def acquire(cache,output,header_seed=None,index_seed=None):
    from PIL import Image
    cache.mkdir(parents=True,exist_ok=True)
    # Seed only exact previously fetched bytes; normal use fetches checked ranges.
    if header_seed or index_seed: raise ValueError('unvalidated seed path unsupported')
    hpath=fetch_range(cache,0,4096)
    if not hpath.read_bytes().startswith(b'#ROSBAG V2.0\n'): raise ValueError('unexpected bag signature')
    ipath=fetch_range(cache,INDEX,TOTAL)
    connections,chunks=index_records(ipath.read_bytes())
    rgb_id=next(k for k,v in connections.items() if v['topic']=='/camera/rgb/image_color')
    depth_id=next(k for k,v in connections.items() if v['topic']=='/camera/depth/image')
    first=next(c for c in chunks if rgb_id in c['connections'])
    first_images=images(fetch_range(cache,first['pos'],first['stop']),connections)
    origin_rgb=min(m['captured_at_s'] for m in first_images if m['topic']=='/camera/rgb/image_color')
    # Select by indexed recorder time before seeing image contents or model output.
    chosen={}
    for offset in OFFSETS:
        target=origin_rgb+offset
        for identifier in (rgb_id,depth_id):
            candidates=[c for c in chunks if identifier in c['connections']]
            candidate=min(candidates,key=lambda c:(abs(c['end']-target),c['pos']))
            chosen[candidate['pos']]=candidate
    print('selected ranges',len(chosen),'bytes',sum(c['stop']-c['pos'] for c in chosen.values()),flush=True)
    dump(cache/'selection-before-decoding.json',dict(origin_rgb_s=origin_rgb,offsets_s=OFFSETS,
         chunks=list(chosen.values()),rule='nearest indexed chunk end per RGB/depth topic, fixed offsets'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        fetched=list(pool.map(lambda c:fetch_range(cache,c['pos'],c['stop']),chosen.values()))
    candidates=[m for p in fetched for m in images(p,connections)]
    groundtruth=cache/'groundtruth.txt'
    if not groundtruth.exists():
        subprocess.run(['curl.exe','--fail','--silent','--show-error','--max-time','90',
            '--max-filesize','200000','--output',str(groundtruth),URL[:-4]+'-groundtruth.txt'],check=True,timeout=100)
    poses=rows(groundtruth,8)
    rgb_rows=sorted((m['captured_at_s'],str(i)) for i,m in enumerate(candidates) if m['topic']=='/camera/rgb/image_color')
    depth_rows=sorted((m['captured_at_s'],str(i)) for i,m in enumerate(candidates) if m['topic']=='/camera/depth/image')
    output.mkdir(parents=True,exist_ok=False)
    (output/'rgb').mkdir(); (output/'depth').mkdir(); (output/'messages').mkdir()
    selections=[]; provenance=[]
    for offset in OFFSETS:
        target=origin_rgb+offset
        try:
            rgb=nearest(rgb_rows,target); depth=nearest(depth_rows,rgb[0]); pose=nearest(poses,rgb[0])
        except ValueError as exc:
            selections.append(dict(offset_s=offset,requested_rgb_at_s=target,error=str(exc))); continue
        rgb_message,depth_message=(candidates[int(r[1])] for r in (rgb,depth))
        rgb_name=f'rgb/{rgb[0]:.9f}.png'; depth_name=f'depth/{depth[0]:.9f}.float32'
        Image.frombytes('RGB',(640,480),rgb_message['pixels'],'raw',rgb_message['encoding'][:3].upper()).save(output/rgb_name)
        with (output/depth_name).open('xb') as stream: stream.write(depth_message['pixels'])
        for label,m in [('rgb',rgb_message),('depth',depth_message)]:
            message_name=f'messages/{offset}-{label}.bin'
            with (output/message_name).open('xb') as stream: stream.write(m['raw_message'])
            provenance.append(dict(offset_s=offset,message_file=message_name,
                **{k:v for k,v in m.items() if k not in ('pixels','raw_message')}))
        selections.append(dict(offset_s=offset,requested_rgb_at_s=target,rgb=(rgb[0],rgb_name),
            depth=(depth[0],depth_name),pose=pose,depth_format='32FC1_LE'))
    (output/'groundtruth.txt').write_bytes(groundtruth.read_bytes())
    dump(output/'selection.json',dict(origin_s=poses[0][0],offsets_s=OFFSETS,frames=selections,
        source_format='official ROS bag subset',selection_rule='fixed bag indexed chunk end; original image header stamps checked within 20 ms'))
    dump(output/'message-provenance.json',provenance)
    dump(output/'sources.json',dict(url=URL,license='CC BY 4.0',
        authors='J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers; TUM RGB-D benchmark',
        license_source='https://cvg.cit.tum.de/data/datasets/rgbd-dataset',
        format_source='https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats',
        groundtruth_url=URL[:-4]+'-groundtruth.txt',
        modifications='Fixed range subset; raw Image messages retained; RGB lossless PNG; optical-Z float32 unchanged',
        cache=str(cache.resolve()),ranges={p.name:sha(p) for p in sorted(cache.glob('*.bin'))},
        full_bag_downloaded=False,tgz_equivalence_checked=False))
    dump(output/'manifest.json',manifest(output))
    return dict(frames=len(selections),matched=sum('error' not in r for r in selections))


def verify_dataset(dataset):
    """Trace every converted sample back to saved official HTTP byte ranges."""
    from PIL import Image
    source=json.loads((dataset/'sources.json').read_text(encoding='utf-8'))
    cache=Path(source['cache'])
    for name,expected in source['ranges'].items():
        if sha(cache/name)!=expected: raise ValueError('original bag range differs')
        start,stop=map(int,Path(name).stem.split('-'))
        text=(cache/f'{start}-{stop}.http').read_text(encoding='ascii').lower()
        if f'content-range: bytes {start}-{stop-1}/{TOTAL}' not in text:
            raise ValueError('saved range provenance differs')
    connections,_=index_records((cache/f'{INDEX}-{TOTAL}.bin').read_bytes())
    provenance=json.loads((dataset/'message-provenance.json').read_text(encoding='utf-8'))
    selection=json.loads((dataset/'selection.json').read_text(encoding='utf-8'))
    poses=rows(dataset/'groundtruth.txt',8)
    decoded={}
    for item in provenance:
        name=item['chunk_file']
        if name not in decoded: decoded[name]=images(cache/name,connections)
        original=next(m for m in decoded[name] if m['topic']==item['topic'] and
                      m['captured_at_s']==item['captured_at_s'])
        if original['raw_message']!=(dataset/item['message_file']).read_bytes():
            raise ValueError('original Image message differs')
        selected=next(r for r in selection['frames'] if r['offset_s']==item['offset_s'])
        if item['topic']=='/camera/rgb/image_color':
            expected=Image.frombytes('RGB',(640,480),original['pixels'],'raw',original['encoding'][:3].upper())
            with Image.open(dataset/selected['rgb'][1]) as image:
                if image.mode!='RGB' or image.size!=(640,480) or image.tobytes()!=expected.tobytes():
                    raise ValueError('converted RGB pixels differ from bag')
        else:
            if (dataset/selected['depth'][1]).read_bytes()!=original['pixels']:
                raise ValueError('float32 depths differ from bag')
        if tuple(selected['pose'])!=nearest(poses,selected['rgb'][0]):
            raise ValueError('selected pose differs from measured ground truth')
        if max(abs(selected[k][0]-selected['rgb'][0]) for k in ('depth','pose'))>.02+1e-9:
            raise ValueError('synchronization bound exceeded')
    return dict(verified=True,original_ranges=len(source['ranges']),original_messages=len(provenance))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=Path('D:/DroneNavTools/tum-rgbd-walking/bag-subset'))
    parser.add_argument('--output',type=Path,default=Path('work/tum-rgbd-bag-input-01'))
    args=parser.parse_args()
    print(json.dumps(acquire(args.cache,args.output)),flush=True)
