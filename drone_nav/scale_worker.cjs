// 来源：本项目原创。固定尺度对照；共享 v0.12 预处理，仅分块尺度不同。
'use strict';
const fs = require('node:fs'), path = require('node:path'), zlib = require('node:zlib');
const { performance } = require('node:perf_hooks');
const old = require('./segmentation_worker.cjs');
const math = require('./segmentation_tiles.cjs');
const arms = [{ key: 'whole', grid: 1 }, { key: 'grid2', grid: 2 }, { key: 'grid4', grid: 4 }];
function write(file, data) { fs.writeFileSync(file, data, { flag: 'wx' }); }
function json(file, value) { write(file, JSON.stringify(value, null, 2)); }

async function main(request) {
  const root = request.root, output = request.output;
  if (request.action === 'replay') {
    for (const item of request.cases) for (const arm of arms) {
      const dir = path.join(output, item.key, arm.key);
      const meta = JSON.parse(fs.readFileSync(path.join(dir, 'worker.json'), 'utf8'));
      const plan = math.tiles(meta.source_width, meta.source_height, meta.width, meta.height, arm.grid);
      if (JSON.stringify(plan) !== JSON.stringify(meta.tiles.map(t => t.window))) throw Error('tile plan differs');
      const pred = new Uint8Array(meta.width * meta.height), coverage = new Uint8Array(pred.length);
      for (const tile of meta.tiles) {
        const buffer = zlib.gunzipSync(fs.readFileSync(path.join(dir, `tile-${tile.window.index}.f32.gz`)));
        const logits = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
        const part = math.classifyTile(logits, tile.dims, tile.window, meta.source_width, meta.source_height, meta.width, meta.height);
        math.stitch(pred, meta.width, meta.height, tile.window, part, coverage);
      }
      if (!coverage.every(v => v === 1) || !Buffer.from(pred).equals(fs.readFileSync(path.join(dir, 'ade.u8'))))
        throw Error('prediction replay differs');
    }
    console.log(JSON.stringify({ verified: true, images: request.cases.length, predictions: request.cases.length * 3 }));
    return;
  }
  const ort = require(path.join(root, 'runtime/node_modules/onnxruntime-node'));
  const { PNG } = require(path.join(root, 'runtime/node_modules/pngjs'));
  const pre = JSON.parse(fs.readFileSync(path.join(root, 'model/preprocessor_config.json'), 'utf8'));
  const tSession = performance.now();
  const session = await ort.InferenceSession.create(path.join(root, 'model/model.onnx'), {
    executionProviders: ['cpu'], intraOpNumThreads: 4, interOpNumThreads: 1,
    executionMode: 'sequential', graphOptimizationLevel: 'all',
  });
  const sessionInit = performance.now() - tSession;
  for (const item of request.cases) {
    const dir = path.join(output, item.key); fs.mkdirSync(dir);
    const tDecode = performance.now();
    const image = PNG.sync.read(fs.readFileSync(path.join(root, item.image)));
    for (let i = 3; i < image.data.length; i += 4) if (image.data[i] !== 255) throw Error('transparent input');
    const decodeMs = performance.now() - tDecode;
    const width = 960, height = Math.round(image.height * width / image.width);
    for (const arm of arms) {
      const armDir = path.join(dir, arm.key); fs.mkdirSync(armDir);
      const started = performance.now();
      const plan = math.tiles(image.width, image.height, width, height, arm.grid);
      const prediction = new Uint8Array(width * height), coverage = new Uint8Array(prediction.length);
      const records = [];
      for (const tile of plan) {
        const t0 = performance.now();
        const view = arm.grid === 1 ? image : math.crop(image, tile);
        const tensor = new ort.Tensor('float32', old.normalize(old.resizeRGB(view, 512, 512), pre), [1, 3, 512, 512]);
        const t1 = performance.now();
        const result = await session.run({ [session.inputNames[0]]: tensor });
        const t2 = performance.now(), logits = result[session.outputNames[0]];
        if (logits.type !== 'float32' || logits.dims.join(',') !== '1,150,128,128') throw Error('unexpected output');
        const part = math.classifyTile(logits.data, logits.dims, tile, image.width, image.height, width, height);
        math.stitch(prediction, width, height, tile, part, coverage);
        const t3 = performance.now();
        write(path.join(armDir, `tile-${tile.index}.f32.gz`), zlib.gzipSync(
          Buffer.from(logits.data.buffer, logits.data.byteOffset, logits.data.byteLength), { level: 1 }));
        records.push({ window: tile, dims: logits.dims, preprocess_ms: t1 - t0, inference_ms: t2 - t1,
          postprocess_ms: t3 - t2, archive_ms: performance.now() - t3 });
      }
      if (!coverage.every(v => v === 1)) throw Error('uncovered pixels');
      write(path.join(armDir, 'ade.u8'), prediction);
      const sum = key => records.reduce((a, b) => a + b[key], 0);
      const meta = { source_width: image.width, source_height: image.height, width, height, tiles: records,
        grid: arm.grid, halo: .125, decode_ms: decodeMs, preprocess_ms: sum('preprocess_ms'),
        inference_ms: sum('inference_ms'), postprocess_ms: sum('postprocess_ms'), archive_ms: sum('archive_ms'),
        core_ms: decodeMs + sum('preprocess_ms') + sum('inference_ms') + sum('postprocess_ms'),
        wall_with_archive_ms: performance.now() - started + decodeMs,
        process_peak_rss_bytes: process.resourceUsage().maxRSS * 1024, session_init_ms: sessionInit,
        runtime: ort.env.versions, node_version: process.version, provider: 'cpu', threads: 4 };
      json(path.join(armDir, 'worker.json'), meta);
      console.log(JSON.stringify({ key: item.key, arm: arm.key, core_ms: meta.core_ms }));
    }
    // 所有三组完成后才读取标注；解码共享、标签共享，计时中已明确分开。
    const label = PNG.sync.read(fs.readFileSync(path.join(root, item.label)));
    if (label.width !== image.width || label.height !== image.height) throw Error('label size differs');
    write(path.join(dir, 'uavid.u8'), old.labelGrid(label, width, height));
    const rgb = old.resizeRGB(image, width, height), rgba = Buffer.alloc(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1]; rgba[i * 4 + 2] = rgb[i * 3 + 2]; rgba[i * 4 + 3] = 255;
    }
    write(path.join(dir, 'rgb.u8'), rgb);
    write(path.join(dir, 'input.png'), PNG.sync.write({ width, height, data: rgba }));
  }
  await session.release();
}

if (require.main === module) main(JSON.parse(fs.readFileSync(process.argv[2] === '-' ? 0 : process.argv[2], 'utf8')))
  .catch(e => { console.error(e.stack); process.exitCode = 1; });
