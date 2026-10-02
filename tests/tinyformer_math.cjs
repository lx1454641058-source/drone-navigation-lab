// 来源：本项目原创。已知坐标、跨类别 top-k、重复 query 与非法输出的独立期望。
'use strict';
const assert=require('node:assert/strict'),{decode,owns}=require('../drone_nav/tinyformer_math.cjs');
const tile={x0:10,y0:20,width:100,height:200,index:0,ex0:10,ey0:20,eval_width:50,eval_height:200};
function data(){const l=new Float32Array(3000).fill(-20),b=new Float32Array(1200);for(let q=0;q<300;q++)b.set([.5,.5,.5,.5],q*4);return {l,b};}
function run(d,threshold=.3){return decode(d.l,[1,300,10],d.b,[1,300,4],tile,threshold);}
let d=data();d.l[0]=0;let r=run(d);assert.equal(r.boxes.length,1);assert.deepEqual([r.boxes[0].x1,r.boxes[0].y1,r.boxes[0].x2,r.boxes[0].y2],[35,70,85,170]);assert.equal(r.boxes[0].score,.5);
d=data();d.l[0]=0;d.l[1]=0;r=run(d);assert.equal(r.boxes.length,2);assert.deepEqual(r.boxes.map(b=>b.visdrone_class),[1,2]);assert.equal(r.boxes[0].query,r.boxes[1].query);
d=data();d.l[0]=0;for(let q=0;q<300;q++)d.l[q*10+2]=2;assert.equal(run(d).boxes.length,0); // 非业务类别占满前 300，不能先过滤类别。
d=data();for(const c of [0,1,3,4,5,8])d.l[c]=0;r=run(d);assert.deepEqual(r.boxes.map(b=>b.group),['person','person','vehicle','vehicle','vehicle','vehicle']);assert.equal(run(d,.5).boxes.length,6);assert.equal(run(d,.5001).boxes.length,0);
d=data();d.l[0]=1000;d.b.set([.5,.5,2,2]);r=run(d);assert.equal(r.boxes[0].score,1);assert.deepEqual([r.boxes[0].x1,r.boxes[0].y1,r.boxes[0].x2,r.boxes[0].y2],[10,20,110,220]);
d=data();d.l[0]=0;d.b.set([2,2,.5,.5]);r=run(d);assert.equal(r.degenerate,1);assert.equal(r.boxes.length,0);
d=data();d.l[0]=NaN;assert.throws(()=>run(d));d=data();d.b[0]=Infinity;assert.throws(()=>run(d));assert.throws(()=>decode(new Float32Array(2),[1,2,10],d.b,[1,300,4],tile));
assert.equal(owns({x1:50,x2:70,y1:20,y2:30},tile),false);assert.equal(owns({x1:49,x2:69,y1:20,y2:30},tile),true);
console.log('TinyFormer decode: coordinates, global top-k, class mapping, ties, thresholds, clipping, invalid values and ownership passed');
