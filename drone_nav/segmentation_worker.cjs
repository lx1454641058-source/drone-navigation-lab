// 来源：本项目原创。仅 CPU 推理；模型只接收 RGB，不接收标注、地图或飞行状态。
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const { performance } = require('node:perf_hooks');

function weights(sourceSize, targetSize) {
  const scale = sourceSize / targetSize;
  const support = Math.max(1, scale);
  return Array.from({ length: targetSize }, (_, i) => {
    const center = (i + .5) * scale - .5;
    const row = [];
    let sum = 0;
    for (let j = Math.max(0, Math.ceil(center - support));
      j <= Math.min(sourceSize - 1, Math.floor(center + support)); j++) {
      const w = Math.max(0, 1 - Math.abs(j - center) / support);
      if (w) { row.push([j, w]); sum += w; }
    }
    return row.map(([j, w]) => [j, w / sum]);
  });
}

function resizeRGB(image, width, height) {
  if (![image.width, image.height, width, height].every(n => Number.isInteger(n) && n > 0) ||
      image.data.length !== image.width * image.height * 4) throw Error('invalid RGB shape');
  const wx = weights(image.width, width), wy = weights(image.height, height);
  const temp = new Uint8Array(width * image.height * 3);
  const result = new Uint8Array(width * height * 3);
  // 两次可分离三角滤波，缩小时扩展滤波范围，减少高频纹理混叠。
  for (let y = 0; y < image.height; y++) for (let x = 0; x < width; x++) {
    for (let c = 0; c < 3; c++) {
      let v = 0;
      for (const [sx, w] of wx[x]) v += image.data[(y * image.width + sx) * 4 + c] * w;
      temp[(y * width + x) * 3 + c] = Math.round(v);
    }
  }
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    for (let c = 0; c < 3; c++) {
      let v = 0;
      for (const [sy, w] of wy[y]) v += temp[(sy * width + x) * 3 + c] * w;
      result[(y * width + x) * 3 + c] = Math.round(v);
    }
  }
  return result;
}

function normalize(rgb, config) {
  if (rgb.length !== 512 * 512 * 3 || config.size.width !== 512 || config.size.height !== 512 ||
      config.do_normalize !== true || config.do_rescale !== true || config.resample !== 2)
    throw Error('unsupported preprocessing configuration');
  const n = rgb.length / 3, out = new Float32Array(rgb.length);
  for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++)
    out[c * n + i] = (rgb[i * 3 + c] * config.rescale_factor - config.image_mean[c]) / config.image_std[c];
  return out;
}

function axis(sourceSize, targetSize) {
  return Array.from({ length: targetSize }, (_, i) => {
    const p = Math.max(0, Math.min(sourceSize - 1, (i + .5) * sourceSize / targetSize - .5));
    return [Math.floor(p), Math.min(sourceSize - 1, Math.floor(p) + 1), p - Math.floor(p)];
  });
}

function argmaxResize(logits, dims, width, height) {
  if (dims.length !== 4 || dims[0] !== 1 || !dims.every(n => Number.isInteger(n) && n > 0) ||
      dims[1] > 256 || logits.length !== dims.reduce((a, b) => a * b, 1))
    throw Error('invalid logits shape');
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1)
    throw Error('invalid output shape');
  if (!logits.every(Number.isFinite)) throw Error('nonfinite logits');
  const [, classes, sh, sw] = dims, plane = sh * sw;
  const xs = axis(sw, width), ys = axis(sh, height);
  const best = new Float32Array(width * height).fill(-Infinity), out = new Uint8Array(best.length);
  // 在分数上插值后选类别；不能先取最大类别再平滑类别编号。
  for (let c = 0; c < classes; c++) for (let y = 0; y < height; y++) {
    const [y0, y1, fy] = ys[y], row0 = c * plane + y0 * sw, row1 = c * plane + y1 * sw;
    for (let x = 0; x < width; x++) {
      const [x0, x1, fx] = xs[x];
      const a = logits[row0 + x0] * (1 - fx) + logits[row0 + x1] * fx;
      const b = logits[row1 + x0] * (1 - fx) + logits[row1 + x1] * fx;
      const v = Math.fround(a * (1 - fy) + b * fy), i = y * width + x;
      if (v > best[i]) { best[i] = v; out[i] = c; }
    }
  }
  return out;
}

function labelGrid(image, width, height) {
  const out = new Uint8Array(width * height);
  // 先检查全幅标注，不能靠降采样恰好漏掉坏值。
  for (let i = 0; i < image.data.length; i += 4) {
    const [r, g, b, a] = image.data.subarray(i, i + 4);
    if (r > 7 || r !== g || r !== b || a !== 255) throw Error('unexpected UAVid indexed mask');
  }
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const sx = Math.min(image.width - 1, Math.floor((x + .5) * image.width / width));
    const sy = Math.min(image.height - 1, Math.floor((y + .5) * image.height / height));
    out[y * width + x] = image.data[(sy * image.width + sx) * 4];
  }
  return out;
}

function write(file, data) { fs.writeFileSync(file, data, { flag: 'wx' }); }

async function main(requestFile) {
  const request = JSON.parse(fs.readFileSync(requestFile === '-' ? 0 : requestFile, 'utf8'));
  const root = request.root, output = request.output;
  if (request.action === 'replay') {
    for (const item of request.cases) {
      const dir = path.join(output, item.key);
      const m = JSON.parse(fs.readFileSync(path.join(dir, 'worker.json'), 'utf8'));
      const buffer = zlib.gunzipSync(fs.readFileSync(path.join(dir, 'logits.f32.gz')));
      const logits = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
      const pred = argmaxResize(logits, m.output_dims, m.width, m.height);
      if (!Buffer.from(pred).equals(fs.readFileSync(path.join(dir, 'ade.u8')))) throw Error('logits replay differs');
    }
    console.log(JSON.stringify({ replay_verified: true, cases: request.cases.length }));
    return;
  }
  const ort = require(path.join(root, 'runtime/node_modules/onnxruntime-node'));
  const { PNG } = require(path.join(root, 'runtime/node_modules/pngjs'));
  const pre = JSON.parse(fs.readFileSync(path.join(root, 'model/preprocessor_config.json'), 'utf8'));
  const started = performance.now();
  const session = await ort.InferenceSession.create(path.join(root, 'model/model.onnx'), {
    executionProviders: ['cpu'], intraOpNumThreads: 4, interOpNumThreads: 1,
    executionMode: 'sequential', graphOptimizationLevel: 'all',
  });
  const sessionMs = performance.now() - started;
  if (session.inputNames.length !== 1 || session.outputNames.length !== 1) throw Error('unexpected model IO');
  for (const item of request.cases) {
    const dir = path.join(output, item.key); fs.mkdirSync(dir);
    const t0 = performance.now();
    const image = PNG.sync.read(fs.readFileSync(path.join(root, item.image)));
    for (let i = 3; i < image.data.length; i += 4)
      if (image.data[i] !== 255) throw Error('transparent RGB input');
    const width = 960, height = Math.round(image.height * width / image.width);
    const rgb512 = resizeRGB(image, 512, 512);
    const tensor = new ort.Tensor('float32', normalize(rgb512, pre), [1, 3, 512, 512]);
    const t1 = performance.now();
    const result = await session.run({ [session.inputNames[0]]: tensor });
    const t2 = performance.now();
    const logits = result[session.outputNames[0]];
    if (logits.type !== 'float32' || logits.dims.join(',') !== '1,150,128,128') throw Error('unexpected SegFormer output');
    const pred = argmaxResize(logits.data, logits.dims, width, height);
    const t3 = performance.now();
    // 标注只在推理结束后读取，用于独立评价，绝不作为模型输入。
    const label = PNG.sync.read(fs.readFileSync(path.join(root, item.label)));
    if (label.width !== image.width || label.height !== image.height) throw Error('image/label size mismatch');
    const truth = labelGrid(label, width, height);
    const rgb = resizeRGB(image, width, height);
    write(path.join(dir, 'rgb.u8'), rgb);
    write(path.join(dir, 'uavid.u8'), truth);
    write(path.join(dir, 'ade.u8'), pred);
    write(path.join(dir, 'logits.f32.gz'), zlib.gzipSync(Buffer.from(logits.data.buffer, logits.data.byteOffset, logits.data.byteLength)));
    const rgba = Buffer.alloc(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1]; rgba[i * 4 + 2] = rgb[i * 3 + 2]; rgba[i * 4 + 3] = 255;
    }
    write(path.join(dir, 'input.png'), PNG.sync.write({ width, height, data: rgba }));
    const metadata = {
      width, height, source_width: image.width, source_height: image.height,
      input_name: session.inputNames[0], output_name: session.outputNames[0], output_dims: logits.dims,
      runtime: ort.env.versions, node_version: process.version, provider: 'cpu', threads: 4,
      session_init_ms: sessionMs, preprocess_ms: t1 - t0, inference_ms: t2 - t1,
      postprocess_ms: t3 - t2, rgb_to_labels_ms: t3 - t0,
      total_with_evaluation_and_files_ms: performance.now() - t0,
    };
    write(path.join(dir, 'worker.json'), JSON.stringify(metadata, null, 2));
    console.log(JSON.stringify({ key: item.key, inference_ms: metadata.inference_ms, rgb_to_labels_ms: metadata.rgb_to_labels_ms }));
  }
  await session.release();
}

module.exports = { weights, resizeRGB, normalize, argmaxResize, labelGrid };
if (require.main === module) main(process.argv[2]).catch(e => { console.error(e.stack); process.exitCode = 1; });
