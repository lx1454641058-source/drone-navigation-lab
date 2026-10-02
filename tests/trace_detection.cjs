// 来源：本项目原创。边界归属、跨块去重和抑制支持框的手算检查。
'use strict';const assert=require('node:assert/strict'),m=require('../tools/trace_detection.cjs');
const b={x1:4,y1:0,x2:6,y2:2,group:'person',score:.9,tile:0,audit_id:'0:0',owned:true};
assert.equal(m.owns(b,{ex0:0,ey0:0,eval_width:5,eval_height:5}),false);
assert.equal(m.owns(b,{ex0:5,ey0:0,eval_width:5,eval_height:5}),true);
const context={...b,score:.95,audit_id:'1:0',owned:false,tile:1};
const duplicate={...b,score:.8,audit_id:'0:1'};
const vehicle={...b,group:'vehicle',audit_id:'0:2'};
const weak={...b,score:.1,audit_id:'0:3'};
const r=m.variants([b,context,duplicate,vehicle,weak]);
assert.deepEqual(r.variants.baseline.map(b=>b.audit_id),['0:0','0:2']);
assert.deepEqual(r.variants.no_core.map(b=>b.audit_id),['1:0','0:2']);
assert.equal(r.variants.no_nms.length,3);assert.equal(r.variants.no_core_no_nms.length,4);
assert.deepEqual(r.suppressions.baseline,[{removed:'0:1',kept:'0:0',iou:1}]);
assert.equal(r.suppressions.no_core.find(s=>s.removed==='0:0').kept,'1:0');
assert.deepEqual(m.variants([b,{...duplicate,score:.9}]).suppressions.baseline,[{removed:'0:1',kept:'0:0',iou:1}]);
console.log('trace detection checks passed');
