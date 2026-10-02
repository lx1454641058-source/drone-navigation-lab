// 来源：本项目原创。整图双模型推理/重放及业务类别去重关系，不修改旧解码器。
'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib'),{performance}=require('node:perf_hooks');
const yolo=require('../drone_nav/detection_math.cjs'),tiny=require('../drone_nav/tinyformer_math.cjs');
const write=(p,v)=>fs.writeFileSync(p,v,{flag:'wx'});
function floats(p){const b=zlib.gunzipSync(fs.readFileSync(p));if(b.length%4)throw Error('unaligned output');return new Float32Array(b.buffer,b.byteOffset,b.length/4);}
function save(p,data){write(p,zlib.gzipSync(Buffer.from(data.buffer,data.byteOffset,data.byteLength)));}
function dedup(boxes){
  const kept=yolo.nms(boxes,.45),keptIds=kept.map(b=>boxes.indexOf(b)),removed=[];
  boxes.forEach((b,i)=>{
    if(keptIds.includes(i))return;
    // 只允许比当前候选优先的框作为抑制证据；同分按输入顺序。
    const by=kept.find(k=>k.group===b.group&&yolo.iou(k,b)>.45&&(k.score>b.score||k.score===b.score&&boxes.indexOf(k)<i));
    if(!by)throw Error('no suppression witness');
    removed.push({prediction:i,kept_prediction:boxes.indexOf(by),iou:yolo.iou(by,b)});
  });
  return {boxes:kept,kept_predictions:keptIds,removed};
}
function windowFor(c){return {index:0,x0:0,y0:0,width:c.width,height:c.height,ex0:0,ey0:0,eval_width:c.width,eval_height:c.height};}
async function main(q){
  if(q.cases.length!==12||new Set(q.cases.map(c=>c.key)).size!==12)throw Error('expected twelve distinct cases');
  if(q.dev){
    for(const c of q.cases){
      const folder=path.join(q.source,c.key,'grid1'),r=JSON.parse(fs.readFileSync(path.join(folder,'result.json'),'utf8'));
      const t=r.tiles[0],decoded=tiny.decode(floats(path.join(folder,'logits-0.gz')),t.logits_dims,floats(path.join(folder,'boxes-0.gz')),t.boxes_dims,t.window);
      if(JSON.stringify(decoded.boxes)!==JSON.stringify(r.boxes))throw Error('development raw replay differs');
      const processed=dedup(decoded.boxes);
      if(q.replay){if(JSON.stringify(processed)!==fs.readFileSync(path.join(q.output,c.key+'.json'),'utf8'))throw Error('development dedup differs');}
      else write(path.join(q.output,c.key+'.json'),JSON.stringify(processed));
    }
    console.log('12 development raw outputs decoded and dedup replayed');return;
  }
  const sessions={};let ort,PNG;
  try{
    if(!q.replay){
      ort=require(path.join(q.runtime,'runtime/node_modules/onnxruntime-node'));PNG=require(path.join(q.runtime,'runtime/node_modules/pngjs')).PNG;
      for(const model of ['yolox','tinyformer'])sessions[model]=await ort.InferenceSession.create(q.models[model],{executionProviders:['cpu'],intraOpNumThreads:4,interOpNumThreads:1,executionMode:'sequential',graphOptimizationLevel:'all'});
      if(sessions.tinyformer.inputNames.join()!=='images'||sessions.tinyformer.outputNames.join()!=='pred_logits,pred_boxes')throw Error('TinyFormer model interface');
    }
    for(const c of q.cases){
      const dir=path.join(q.output,c.key),old=q.replay?JSON.parse(fs.readFileSync(path.join(dir,'result.json'),'utf8')):null;
      const t=windowFor(c),ratio=Math.min(640/c.width,640/c.height),prep={ratio,width:Math.floor(c.width*ratio),height:Math.floor(c.height*ratio)};
      let yd,ld,bd,ys,ls,bs,measurements;
      if(q.replay){
        ({yolox:yd,logits:ld,coords:bd}=old.dimensions);
        ys=floats(path.join(dir,'yolox-output.gz'));ls=floats(path.join(dir,'tiny-logits.gz'));bs=floats(path.join(dir,'tiny-boxes.gz'));
      }else{
        const im=PNG.sync.read(fs.readFileSync(path.join(dir,'input.png')));
        if(im.width!==c.width||im.height!==c.height)throw Error('image dimensions');
        const start=performance.now(),yp=yolo.prepare(im),prepared=performance.now();
        const yo=await sessions.yolox.run({[sessions.yolox.inputNames[0]]:new ort.Tensor('float32',yp.values,[1,3,640,640])});
        const inferred=performance.now(),y=yo[sessions.yolox.outputNames[0]];
        if(y.type!=='float32')throw Error('YOLOX dtype');
        ys=y.data;yd=y.dims;save(path.join(dir,'yolox-input.gz'),yp.values);save(path.join(dir,'yolox-output.gz'),ys);
        const ti=floats(path.join(dir,'tiny-input.gz'));if(ti.length!==3*640*640||!ti.every(Number.isFinite))throw Error('TinyFormer input');
        const tstart=performance.now(),to=await sessions.tinyformer.run({images:new ort.Tensor('float32',ti,[1,3,640,640])});
        const tend=performance.now();if(to.pred_logits.type!=='float32'||to.pred_boxes.type!=='float32')throw Error('TinyFormer dtype');
        ls=to.pred_logits.data;ld=to.pred_logits.dims;bs=to.pred_boxes.data;bd=to.pred_boxes.dims;
        save(path.join(dir,'tiny-logits.gz'),ls);save(path.join(dir,'tiny-boxes.gz'),bs);
        measurements={yolox_preprocess_ms:prepared-start,yolox_inference_ms:inferred-prepared,tinyformer_inference_ms:tend-tstart};
      }
      const start=performance.now(),yb=yolo.nms(yolo.decode(ys,yd,prep,t)),tb=tiny.decode(ls,ld,bs,bd,t);
      const beforeDedup=performance.now(),clean=dedup(tb.boxes),end=performance.now();
      const decoded={yolox:yb,tinyformer:tb.boxes,tinyformer_nms:clean,degenerate:tb.degenerate};
      if(q.replay){if(JSON.stringify(decoded)!==JSON.stringify(old.decoded))throw Error('holdout decoded result differs');}
      else write(path.join(dir,'result.json'),JSON.stringify({dimensions:{yolox:yd,logits:ld,coords:bd},decoded,measurements:{...measurements,decode_ms:beforeDedup-start,dedup_ms:end-beforeDedup}},null,2));
      console.log(c.key,q.replay?'replayed':'inferred');
    }
  }finally{for(const s of Object.values(sessions))await s.release();}
  console.log('24 whole-image model calls',q.replay?'replayed':'saved');
}
module.exports={dedup,windowFor};
if(require.main===module)main(JSON.parse(fs.readFileSync(0,'utf8'))).catch(e=>{console.error(e.stack);process.exitCode=1;});
