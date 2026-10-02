// 来源：本项目原创。独立手算去重边界与抑制关系。
'use strict';
const assert=require('node:assert/strict'),{dedup}=require('../tools/whole_detection_worker.cjs');
const box=(x1=0,x2=10,score=.9,group='person')=>({x1,y1:0,x2,y2:10,score,group});
assert.deepEqual(dedup([]),{boxes:[],kept_predictions:[],removed:[]});
const overlap=[box(),box(0,10,.8),box(0,10,.7,'vehicle')];
let r=dedup(overlap);
assert.deepEqual(r.kept_predictions,[0,2]);
assert.deepEqual(r.removed,[{prediction:1,kept_prediction:0,iou:1}]);
assert.equal(overlap.length,3);
assert.deepEqual(dedup([box(0,10,.8),box(0,10,.9)]).kept_predictions,[1]);
assert.deepEqual(dedup([box(),box()]).kept_predictions,[0]);
// A 抑制 B，B 与 C 重叠但 A 与 C 不重叠到门槛；已移除的 B 不能继续抑制 C。
assert.deepEqual(dedup([box(0,10,.9),box(3,13,.8),box(6,16,.7)]).kept_predictions,[0,2]);
// 被包含小框面积 / 大框面积 = 9 / 20，正好 0.45，不抑制。
assert.deepEqual(dedup([box(0,20),box(0,9,.8)]).kept_predictions,[0,1]);
assert.deepEqual(dedup([box(0,20),box(0,9.001,.8)]).kept_predictions,[0]);
assert.throws(()=>dedup([box(0,0)]),/invalid box/);
assert.throws(()=>dedup([box(0,10,NaN)]),/invalid box/);
console.log('dedup boundary, stable order, cross-group and suppression witness checks passed');
