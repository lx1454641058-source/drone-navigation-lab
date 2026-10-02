"""来源：本项目原创。按公开格式核对固定 SegFormer 导出参数；不执行 pickle 或模型代码。"""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import urllib.request

ONNX_SHA = '3e5c18a4be395f16646438d54c42377ddc202edfa33d5eced0c9506de75c44c2'
SAFE_SHA = '6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9'
UPSTREAM = '489d5cd81a0b59fab9b7ea758d3548ebe99677da'
SAFE_URL = f'https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/{UPSTREAM}/model.safetensors'


def varint(blob, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(blob):
            raise ValueError('truncated varint')
        byte = blob[position]
        position += 1
        if shift == 63 and byte > 1:
            raise ValueError('varint exceeds 64 bits')
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, position
    raise ValueError('oversized varint')


def fields(blob):
    """只读 Protobuf 的标量和字节字段；未知编码直接拒绝，不构造可执行对象。"""
    position = 0
    while position < len(blob):
        key, position = varint(blob, position)
        tag, wire = key >> 3, key & 7
        if not tag:
            raise ValueError('invalid field tag')
        if wire == 0:
            value, position = varint(blob, position)
        else:
            if wire in (1, 5):
                size = 8 if wire == 1 else 4
            elif wire == 2:
                size, position = varint(blob, position)
            else:
                raise ValueError('unsupported protobuf wire')
            if size > len(blob) - position:
                raise ValueError('truncated field')
            value = blob[position:position + size]
            position += size
        yield tag, value


def floats(blob):
    values = array('f')
    values.frombytes(blob)
    if sys.byteorder != 'little':
        values.byteswap()
    return values


def transpose(blob, rows, columns):
    values = floats(blob)
    if rows < 1 or columns < 1 or len(values) != rows * columns:
        raise ValueError('invalid transpose dimensions')
    result = array('f', (values[i * columns + j] for j in range(columns) for i in range(rows)))
    if sys.byteorder != 'little':
        result.byteswap()
    return result.tobytes()


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def inspect(onnx_path, safe_path):
    onnx = onnx_path.read_bytes()
    safe = safe_path.read_bytes()
    if hashlib.sha256(onnx).hexdigest() != ONNX_SHA or hashlib.sha256(safe).hexdigest() != SAFE_SHA:
        raise ValueError('fixed model checksum differs')
    graph = next(value for tag, value in fields(onnx) if tag == 7)
    parameters = {}
    for tag, tensor in fields(graph):
        if tag != 5:
            continue
        entries = list(fields(tensor))
        name = next(v.decode('utf-8') for k, v in entries if k == 8)
        dtype = next(v for k, v in entries if k == 2)
        raw = next(v for k, v in entries if k == 9)
        if dtype != 1 or name in parameters:
            raise ValueError('unexpected initializer dtype/name')
        parameters[name] = raw
    size = struct.unpack('<Q', safe[:8])[0]
    header = json.loads(safe[8:8 + size])
    data = safe[8 + size:]

    def body(name):
        a, b = header[name]['data_offsets']
        if not 0 <= a <= b <= len(data):
            raise ValueError('safetensors offset outside data')
        return data[a:b]

    transformed = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        blob = body(name)
        transformed[hashlib.sha256(blob).hexdigest()] = (name, 'identical')
        if len(meta['shape']) == 2 and meta['dtype'] == 'F32':
            transposed = transpose(blob, *meta['shape'])
            transformed[hashlib.sha256(transposed).hexdigest()] = (name, 'transpose')
    exact, renamed, remaining = [], [], []
    for name, raw in parameters.items():
        if name in header:
            if raw != body(name):
                raise ValueError('named parameter differs: ' + name)
            exact.append(name)
        else:
            match = transformed.get(hashlib.sha256(raw).hexdigest())
            if match:
                renamed.append(dict(onnx=name, upstream=match[0], operation=match[1]))
            else:
                remaining.append(name)
    if set(remaining) != {'onnx::Conv_1780', 'onnx::Conv_1781'}:
        raise ValueError('unexpected transformed constants')
    # 标准 Conv + BatchNorm 推理融合公式；按 float32 保存中间结果，仅容许小数值舍入差。
    bn = 'decode_head.batch_norm.'
    gamma, beta, mean, variance = [floats(body(bn + name)) for name in
                                   ('weight', 'bias', 'running_mean', 'running_var')]
    weights = floats(body('decode_head.linear_fuse.weight'))
    shape = header['decode_head.linear_fuse.weight']['shape']
    stride = math.prod(shape[1:])
    fused_weights, fused_bias = [], []
    for i in range(shape[0]):
        scale = f32(gamma[i] / math.sqrt(f32(variance[i] + 1e-5)))
        fused_weights.extend(f32(v * scale) for v in weights[i * stride:(i + 1) * stride])
        fused_bias.append(f32(beta[i] - f32(mean[i] * scale)))
    fusion = []
    for name, expected in [('onnx::Conv_1780', fused_weights), ('onnx::Conv_1781', fused_bias)]:
        actual = floats(parameters[name])
        if len(actual) != len(expected):
            raise ValueError('fused shape differs')
        differences = [abs(a - b) for a, b in zip(actual, expected)]
        maximum = max(differences)
        if not math.isfinite(maximum) or maximum > 1e-6:
            raise ValueError('fused parameter differs')
        fusion.append(dict(name=name, values=len(actual), maximum_abs_difference=maximum,
                           mean_abs_difference=sum(differences) / len(differences)))
    if (len(exact), len(renamed), len(fusion)) != (150, 52, 2):
        raise ValueError('parameter coverage differs')
    return dict(verified=True, onnx_sha256=ONNX_SHA, safetensors_sha256=SAFE_SHA,
                upstream_revision=UPSTREAM, exact_named=exact, transformed=renamed, fusion=fusion,
                scope='204 个导出参数核对；不证明计算图、运行算子和整个网络输出逐位一致')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='只读核对固定 ONNX 与 NVIDIA 原始参数')
    parser.add_argument('--onnx', type=Path, default=Path('D:/DroneNavTools/vision-v12/model/model.onnx'))
    parser.add_argument('--safetensors', type=Path, default=Path('D:/DroneNavTools/work/vision-v13/model.safetensors'))
    parser.add_argument('--output', type=Path, required=True, help='新建报告，已有文件拒绝覆盖')
    parser.add_argument('--download-reference', action='store_true', help='缺少上游参数时从固定来源下载约 15 MB；限 NVIDIA 研究许可')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refuse to overwrite existing inspection')
    if args.download_reference and not args.safetensors.exists():
        with urllib.request.urlopen(SAFE_URL, timeout=60) as response:
            reference = response.read(15036945)
        if len(reference) != 15036944 or hashlib.sha256(reference).hexdigest() != SAFE_SHA:
            raise ValueError('downloaded reference checksum differs')
        args.safetensors.parent.mkdir(parents=True, exist_ok=True)
        with args.safetensors.open('xb') as stream:
            stream.write(reference)
    result = inspect(args.onnx, args.safetensors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print('verified', len(result['exact_named']), len(result['transformed']), len(result['fusion']))
