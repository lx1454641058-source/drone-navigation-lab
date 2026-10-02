"""来源：本项目原创。固定许可候选准备、开发探针与三类像素评价；不改旧实验。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from drone_nav.realvision import asset_root, validate_assets, dump, sha
from tools.setup_vision import put

MODEL_ROOT = Path('D:/DroneNavTools/vision-cityscapes-probe')
REPO = 'Xenova/segformer-b0-finetuned-cityscapes-640-1280'
REV = 'ae0e7d35a49bb915f33a58b2ea93f4a6323bcb3f'
NV_REPO = 'nvidia/segformer-b0-finetuned-cityscapes-640-1280'
NV_REV = '618918f3e955c8c4364d73cdbd403a40282b98b9'
# 小文本用官方 API 返回的 Git blob ID 核对；模型使用 LFS SHA-256。
FILES = (
    ('model.onnx', REPO, REV, 'onnx/model.onnx', 15200774, 'sha256', '4dc3ab8ed0c6d724ad58fa7ddfeea0f38c85f2e5c0aa544c88a83bc233f95292'),
    ('config.json', REPO, REV, 'config.json', 1761, 'git', 'e2e2c8c411a1c7116aa8a5fc9916c9b1ec3d6a2d'),
    ('preprocessor_config.json', REPO, REV, 'preprocessor_config.json', 374, 'git', '89faa86b52097b90ef95c2cc85eb6c298a24a57e'),
    ('README.md', REPO, REV, 'README.md', 1697, 'git', 'ab3f07c734417c24d7487d37ec10b6fc7ec348dd'),
    ('NVIDIA-README.md', NV_REPO, NV_REV, 'README.md', 3133, 'git', '7bdf55627b477393b53f71cb52f6c3479a9b953d'),
)
LABELS = ('road','sidewalk','building','wall','fence','pole','traffic light','traffic sign',
          'vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle')
GROUPS = {'road': (2, (0,)), 'vehicle': (3, (13,14,15)), 'person': (6, (11,))}


def binary_metrics(truth, prediction, truth_id, predicted_ids):
    if not truth or len(truth) != len(prediction):
        raise ValueError('empty or mismatched maps')
    tp = fp = fn = 0
    for actual, estimate in zip(truth, prediction):
        a, p = actual == truth_id, estimate in predicted_ids
        tp += a and p
        fp += not a and p
        fn += a and not p
    return dict(tp=tp, fp=fp, fn=fn, iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                recall=tp/(tp+fn) if tp+fn else None, precision=tp/(tp+fp) if tp+fp else None)


def prepare(root):
    records = []
    for name, repo, rev, remote, size, kind, expected in FILES:
        target = root/name
        url = f'https://huggingface.co/{repo}/resolve/{rev}/{remote}'
        if target.exists():
            body = target.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=90) as response:
                body = response.read(size+1)
        digest = hashlib.sha256(body).hexdigest() if kind == 'sha256' else hashlib.sha1(
            f'blob {len(body)}\0'.encode()+body).hexdigest()
        if len(body) != size or digest != expected:
            raise ValueError('candidate resource digest differs: '+name)
        put(target, body)
        records.append(dict(path=name,url=url,bytes=size,sha256=sha(target)))
    license_file = asset_root()/'licenses/SegFormer-LICENSE'
    if sha(license_file) != 'f549820c06519e3105e5174a2fd7285224ffd2268a7f879a843b1a255fd04a61':
        raise ValueError('license differs')
    put(root/'LICENSE',license_file.read_bytes())
    records.append(dict(path='LICENSE',url='https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/LICENSE',
                        bytes=license_file.stat().st_size,sha256=sha(license_file)))
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    if config['id2label'] != {str(i):v for i,v in enumerate(LABELS)} or config['hidden_sizes'] != [32,64,160,256]:
        raise ValueError('candidate labels/architecture differ')
    return records


def run(output, reference):
    if output.exists():
        raise FileExistsError('refuse to overwrite: '+str(output))
    root=asset_root().resolve()
    lock,runtime=validate_assets(root)
    records=prepare(MODEL_ROOT)
    cases=[c for c in lock['cases'] if c['split']=='development']
    manifest=json.loads((reference/'manifest.json').read_text(encoding='utf-8'))
    refs={}
    for c in cases:
        for suffix in ('truth.u8','whole/prediction.u8','grid4/prediction.u8'):
            name=c['key']+'/'+suffix
            if sha(reference/name) != manifest[name]:
                raise ValueError('baseline changed')
            refs[name]=manifest[name]
    output.mkdir(parents=True,exist_ok=False)
    protocol=PROJECT/'docs/CITYSCAPES_PROBE_PROTOCOL.md'
    put(output/'protocol.md',protocol.read_bytes())
    dump(output/'sources.json',records)
    dump(output/'runtime-lock.json',runtime)
    worker=Path(__file__).with_suffix('.cjs')
    source_paths=[worker,PROJECT/'drone_nav/segmentation_worker.cjs',PROJECT/'drone_nav/segmentation_tiles.cjs',Path(__file__)]
    source_hashes={p.name:sha(p) for p in source_paths}
    q=dict(root=str(root),model=str(MODEL_ROOT.resolve()),output=str(output.resolve()),cases=cases)
    dump(output/'request.json',q)
    subprocess.run(['node',str(worker)],input=json.dumps(q),text=True,check=True,timeout=600)
    results=[]
    for c in cases:
        truth=(reference/c['key']/'truth.u8').read_bytes()
        put(output/c['key']/'truth.u8',truth)
        for grid,old in [(1,'whole'),(4,'grid4')]:
            pred=(output/c['key']/f'grid{grid}'/'city.u8').read_bytes()
            if any(v>=19 for v in pred):
                raise ValueError('invalid city class')
            baseline=(reference/c['key']/old/'prediction.u8').read_bytes()
            scores={name:dict(city=binary_metrics(truth,pred,tid,pids),ade=binary_metrics(truth,baseline,tid,(tid,)))
                    for name,(tid,pids) in GROUPS.items()}
            results.append(dict(key=c['key'],grid=grid,metrics=scores))
    subprocess.run(['node',str(worker)],input=json.dumps(dict(q,replay=True)),text=True,check=True,timeout=600)
    if {p.name:sha(p) for p in source_paths} != source_hashes:
        raise ValueError('source changed during run')
    dump(output/'report.json',dict(kind='development_candidate_only',model_repository=REPO,model_revision=REV,
        labels=LABELS,groups=GROUPS,reference=str(reference.resolve()),reference_hashes=refs,source_hashes=source_hashes,
        protocol_sha256=sha(protocol),results=results,replay_passed=True))
    dump(output/'manifest.json',{str(p.relative_to(output)).replace('\\','/'):sha(p) for p in output.rglob('*') if p.is_file()})
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description='仅两个开发帧的 Cityscapes 候选探针，新增约15 MB研究权重')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=PROJECT/'work/run-v13-scale-validated')
    a=p.parse_args()
    print(json.dumps(run(a.output,a.reference),ensure_ascii=False,indent=2))
