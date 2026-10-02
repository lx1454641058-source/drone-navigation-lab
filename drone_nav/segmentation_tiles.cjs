// 来源：本项目原创。固定分块及带上下文的像素采样；旧整图实现保留为对照。
'use strict';

function integer(n, name) {
  if (!Number.isInteger(n) || n < 1 || n > 16384) throw Error('invalid ' + name);
}

function tiles(width, height, evalWidth, evalHeight, grid, halo = .125) {
  for (const [n, name] of [[width, 'width'], [height, 'height'], [evalWidth, 'eval width'], [evalHeight, 'eval height']]) integer(n, name);
  if (![1, 2, 4].includes(grid) || !Number.isFinite(halo) || halo < 0 || halo > .5 ||
      Math.min(width, height, evalWidth, evalHeight) < grid) throw Error('invalid tile grid');
  const result = [];
  for (let row = 0; row < grid; row++) for (let col = 0; col < grid; col++) {
    // 先分配评价像素的唯一归属；源图窗口涵盖核心区域及固定比例的上下文。
    const ex0 = Math.floor(col * evalWidth / grid), ex1 = Math.floor((col + 1) * evalWidth / grid);
    const ey0 = Math.floor(row * evalHeight / grid), ey1 = Math.floor((row + 1) * evalHeight / grid);
    const cx0 = ex0 * width / evalWidth, cx1 = ex1 * width / evalWidth;
    const cy0 = ey0 * height / evalHeight, cy1 = ey1 * height / evalHeight;
    const hx = (cx1 - cx0) * halo, hy = (cy1 - cy0) * halo;
    const x0 = Math.max(0, Math.floor(cx0 - hx)), x1 = Math.min(width, Math.ceil(cx1 + hx));
    const y0 = Math.max(0, Math.floor(cy0 - hy)), y1 = Math.min(height, Math.ceil(cy1 + hy));
    result.push({ index: result.length, row, col, x0, y0, width: x1 - x0, height: y1 - y0,
      ex0, ey0, eval_width: ex1 - ex0, eval_height: ey1 - ey0 });
  }
  return result;
}

function crop(image, tile) {
  if (!image.data || image.data.length !== image.width * image.height * 4 ||
      tile.x0 < 0 || tile.y0 < 0 || tile.x0 + tile.width > image.width || tile.y0 + tile.height > image.height)
    throw Error('invalid crop');
  const data = Buffer.alloc(tile.width * tile.height * 4);
  for (let y = 0; y < tile.height; y++) {
    const start = ((y + tile.y0) * image.width + tile.x0) * 4;
    data.set(image.data.subarray(start, start + tile.width * 4), y * tile.width * 4);
  }
  return { width: tile.width, height: tile.height, data };
}

function samplingAxis(logitSize, sourceSize, cropStart, cropSize, evalStart, evalCount, evalSize) {
  const low = new Int32Array(evalCount), high = new Int32Array(evalCount), fraction = new Float64Array(evalCount);
  for (let i = 0; i < evalCount; i++) {
    // 把评价像素中心映回原照片，再映入当前分块；保留半像素约定，避免奇数尺寸拼接偏移。
    const p = cropStart === 0 && cropSize === sourceSize ?
      (i + evalStart + .5) * logitSize / evalSize - .5 :
      (((i + evalStart + .5) * sourceSize / evalSize - cropStart) * logitSize / cropSize - .5);
    const bounded = Math.max(0, Math.min(logitSize - 1, p));
    low[i] = Math.floor(bounded); high[i] = Math.min(logitSize - 1, low[i] + 1); fraction[i] = bounded - low[i];
  }
  return { low, high, fraction };
}

function classifyTile(logits, dims, tile, sourceWidth, sourceHeight, evalWidth, evalHeight) {
  if (dims.length !== 4 || dims[0] !== 1 || !dims.every(n => Number.isInteger(n) && n > 0) || dims[1] > 256 ||
      logits.length !== dims.reduce((a, b) => a * b, 1) || !logits.every(Number.isFinite)) throw Error('invalid logits');
  const [, classes, sh, sw] = dims, w = tile.eval_width, h = tile.eval_height;
  integer(w, 'tile eval width'); integer(h, 'tile eval height');
  const xs = samplingAxis(sw, sourceWidth, tile.x0, tile.width, tile.ex0, w, evalWidth);
  const ys = samplingAxis(sh, sourceHeight, tile.y0, tile.height, tile.ey0, h, evalHeight);
  const best = new Float32Array(w * h).fill(-Infinity), prediction = new Uint8Array(w * h), plane = sh * sw;
  // 预先展开坐标；热点循环不创建数组、不解构，保持与旧整图分数计算相同的运算顺序。
  for (let c = 0; c < classes; c++) for (let y = 0; y < h; y++) {
    const row0 = c * plane + ys.low[y] * sw, row1 = c * plane + ys.high[y] * sw, fy = ys.fraction[y];
    const start = y * w;
    for (let x = 0; x < w; x++) {
      const x0 = xs.low[x], x1 = xs.high[x], fx = xs.fraction[x];
      const a = logits[row0 + x0] * (1 - fx) + logits[row0 + x1] * fx;
      const b = logits[row1 + x0] * (1 - fx) + logits[row1 + x1] * fx;
      const v = Math.fround(a * (1 - fy) + b * fy), i = start + x;
      if (v > best[i]) { best[i] = v; prediction[i] = c; }
    }
  }
  return prediction;
}

function stitch(output, width, height, tile, prediction, coverage) {
  if (output.length !== width * height || coverage.length !== output.length ||
      prediction.length !== tile.eval_width * tile.eval_height || tile.ex0 < 0 || tile.ey0 < 0 ||
      tile.ex0 + tile.eval_width > width || tile.ey0 + tile.eval_height > height) throw Error('invalid stitch');
  for (let y = 0; y < tile.eval_height; y++) for (let x = 0; x < tile.eval_width; x++) {
    const index = (tile.ey0 + y) * width + tile.ex0 + x;
    if (coverage[index]) throw Error('tile core overlaps');
    output[index] = prediction[y * tile.eval_width + x]; coverage[index] = 1;
  }
}

module.exports = { tiles, crop, samplingAxis, classifyTile, stitch };
