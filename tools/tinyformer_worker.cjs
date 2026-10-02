// 来源：本项目原创。只读取本地固定 ONNX、预处理张量和请求，保存两路原始输出。
'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const {performance}=require('node:perf_hooks');
const {decode,owns}=require('../drone_nav/tinyformer_math.cjs'),{nms}=require('../drone_nav/detection_math.cjs');
const write=(p,v)=>fs.writeFileSync(p,v,{flag:'wx'});
const readFloats=p=>{const b=zlib.gunzipSync(fs.readFileSync(p));if(b.length%4)throw Error('unaligned tensor');return new Float32Array(b.buffer,b.byteOffset,b.length/4);};
async function main(q){
  if(q.cases.length!==12||new Set(q.cases.map(c=>c.key)).size!==12)throw Error('fixed twelve frames required');
  let session,ort,calls=0;
  try{
    if(!q.replay){
      ort=require(path.join(q.runtime,'runtime/node_modules/onnxruntime-node'));
      session=await ort.InferenceSession.create(q.model,{executionProviders:['cpu'],intraOpNumThreads:4,interOpNumThreads:1,executionMode:'sequential',graphOptimizationLevel:'all'});
      if(session.inputNames.join()!=='images'||session.outputNames.join()!=='pred_logits,pred_boxes')throw Error('model interface differs');
    }
    for(const c of q.cases){
      for(const grid of [1,4]){
        const dir=path.join(q.output,c.key,`grid${grid}`),records=[],boxes=[];
        const previous=q.replay?JSON.parse(fs.readFileSync(path.join(dir,'result.json'),'utf8')):null;
        let degenerate=0;
        for(const t of c.plans[String(grid)]){
          let logits,coords,ldims,bdims;
          if(q.replay){
            const old=previous.tiles[t.index];
            if(JSON.stringify(old.window)!==JSON.stringify(t))throw Error('tile window differs');
            ldims=old.logits_dims;bdims=old.boxes_dims;
            logits=readFloats(path.join(dir,`logits-${t.index}.gz`));coords=readFloats(path.join(dir,`boxes-${t.index}.gz`));
          }else{
            const tensor=readFloats(path.join(dir,`input-${t.index}.gz`));
            if(tensor.length!==3*640*640||!tensor.every(Number.isFinite))throw Error('input tensor');
            const start=performance.now(),outputs=await session.run({images:new ort.Tensor('float32',tensor,[1,3,640,640])});
            const elapsed=performance.now()-start;
            const l=outputs.pred_logits,b=outputs.pred_boxes;
            if(l.type!=='float32'||b.type!=='float32')throw Error('output dtype');
            logits=l.data;coords=b.data;ldims=l.dims;bdims=b.dims;
            for(const [name,data]of [['logits',logits],['boxes',coords]])write(path.join(dir,`${name}-${t.index}.gz`),zlib.gzipSync(Buffer.from(data.buffer,data.byteOffset,data.byteLength)));
            records.push({window:t,logits_dims:ldims,boxes_dims:bdims,inference_ms:elapsed});
          }
          const result=decode(logits,ldims,coords,bdims,t);degenerate+=result.degenerate;
          boxes.push(...result.boxes.filter(b=>grid===1||owns(b,t)));calls++;
        }
        const kept=grid===1?boxes:nms(boxes);
        if(q.replay){if(JSON.stringify(kept)!==JSON.stringify(previous.boxes)||degenerate!==previous.degenerate)throw Error('decoded results differ');}
        else write(path.join(dir,'result.json'),JSON.stringify({grid,boxes:kept,degenerate,tiles:records},null,2));
      }
      console.log(c.key,q.replay?'replayed':'inferred');
    }
  }finally{if(session)await session.release();}
  console.log(calls,'TinyFormer outputs',q.replay?'replayed':'saved');
}
if(require.main===module)main(JSON.parse(fs.readFileSync(0,'utf8'))).catch(e=>{console.error(e.stack);process.exitCode=1;});
