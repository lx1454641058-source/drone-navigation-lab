// 来源：本项目原创。与独立旧算法对照，检验奇数尺寸、边界归属和半像素位置。
'use strict';
const assert = require('node:assert/strict');
const old = require('../drone_nav/segmentation_worker.cjs');
const math = require('../drone_nav/segmentation_tiles.cjs');
let randomState = 842;
function random() { randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0; return randomState / 4294967296; }
for (const [w, h] of [[1, 1], [9, 7], [64, 37], [960, 506]]) {
  const dims = [1, 7, 5, 9], logits = Float32Array.from({ length: 315 }, () => random() * 8 - 4);
  const t = math.tiles(4096, 2160, w, h, 1)[0];
  assert.deepEqual(math.classifyTile(logits, dims, t, 4096, 2160, w, h), old.argmaxResize(logits, dims, w, h));
}
for (const grid of [1, 2, 4]) {
  const w = 13, h = 9, list = math.tiles(53, 31, w, h, grid), seen = new Uint8Array(w * h), out = new Uint8Array(w * h);
  for (const t of list) {
    assert(t.x0 >= 0 && t.y0 >= 0 && t.x0 + t.width <= 53 && t.y0 + t.height <= 31);
    math.stitch(out, w, h, t, new Uint8Array(t.eval_width * t.eval_height).fill(t.index), seen);
  }
  assert(seen.every(v => v === 1));
  const t = list[0]; assert.throws(() => math.stitch(out, w, h, t, new Uint8Array(t.eval_width * t.eval_height), seen), /overlaps/);
}
const xs = math.samplingAxis(4, 100, 20, 40, 2, 4, 10);
assert.deepEqual([...xs.low], [0, 1, 2, 3]); assert.deepEqual([...xs.fraction], [0, 0, 0, 0]);
const rgba = Uint8Array.from({ length: 4 * 3 * 4 }, (_, i) => i);
const cut = math.crop({ width: 4, height: 3, data: rgba }, { x0: 1, y0: 1, width: 2, height: 2 });
assert.deepEqual([...cut.data], [...rgba.slice(20, 28), ...rgba.slice(36, 44)]);
assert.throws(() => math.tiles(10, 10, 9, 9, 3), /grid/);
assert.throws(() => math.tiles(10, 10, 9, 9, 2, NaN), /grid/);
assert.throws(() => math.classifyTile(new Float32Array([NaN]), [1, 1, 1, 1], math.tiles(10, 10, 10, 10, 1)[0], 10, 10, 10, 10), /logits/);
console.log('tile math passed');
