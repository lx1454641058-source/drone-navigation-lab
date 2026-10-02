// 来源：本项目原创。用手算的小图检验插值顺序、归一化、透明/未知标注处理。
'use strict';
const assert = require('node:assert/strict');
const { resizeRGB, normalize, argmaxResize, labelGrid } = require('../drone_nav/segmentation_worker.cjs');
const rgb={width:2,height:1,data:Uint8Array.from([0,10,20,255,100,110,120,255])};
assert.deepEqual([...resizeRGB(rgb,1,1)],[50,60,70]);
assert.deepEqual([...resizeRGB(rgb,2,1)],[0,10,20,100,110,120]);
// 分数插值的中间像素选第一类：若先 argmax 再最近邻，会错误选第二类。
assert.deepEqual([...argmaxResize(Float32Array.from([4,0,0,2]),[1,2,1,2],3,1)],[0,0,1]);
assert.deepEqual([...argmaxResize(Float32Array.from([1,1]),[1,2,1,1],2,2)],[0,0,0,0]);
assert.throws(()=>argmaxResize(Float32Array.from([NaN,1]),[1,2,1,1],1,1),/nonfinite/);
assert.throws(()=>argmaxResize(Float32Array.from([1]),[1,2,1,1],1,1),/shape/);
const labels={width:2,height:1,data:Uint8Array.from([1,1,1,255,7,7,7,255])};
assert.deepEqual([...labelGrid(labels,4,1)],[1,1,7,7]);
labels.data[0]=255;assert.throws(()=>labelGrid(labels,1,1),/mask/);
labels.data[0]=1;labels.data[3]=0;assert.throws(()=>labelGrid(labels,1,1),/mask/);
const input=new Uint8Array(512*512*3).fill(255);
const pre={size:{width:512,height:512},do_normalize:true,do_rescale:true,resample:2,
rescale_factor:1/255,image_mean:[.5,.25,0],image_std:[.5,.25,1]};
const tensor=normalize(input,pre);
assert.equal(tensor[0],1);assert.equal(tensor[512*512],3);assert.equal(tensor[2*512*512],1);
console.log('image math passed');
