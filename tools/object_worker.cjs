// 来源：本项目原创。不同图像尺寸下的固定目标检测对照；输出可重放，不接飞控。
'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const {performance}=require('node:perf_hooks');
const math=require('../drone_nav/detection_math.cjs'),tileMath=require('../drone_nav/segmentation_tiles.cjs');
const write=(p,v)=>fs.writeFileSync(p,v,{flag:'wx'});
function boxesFor(scores,dims,t){
  const ratio=Math.min(640/t.width,640/t.height);
  return math.decode(scores,dims,{ratio,width:Math.floor(t.width*ratio),height:Math.floor(t.height*ratio)},t).filter(b=>{
    const x=(b.x1+b.x2)/2,y=(b.y1+b.y2)/2;
    return x>=t.ex0&&x<t.ex0+t.eval_width&&y>=t.ey0&&y<t.ey0+t.eval_height;
  });
}
async function main(q){
  if(q.cases.length!==12||new Set(q.cases.map(c=>c.key)).size!==12)throw Error('fixed twelve cases required');
  let session,ort,PNG;
  if(!q.replay){
    ort=require(path.join(q.runtime,'runtime/node_modules/onnxruntime-node'));
    PNG=require(path.join(q.runtime,'runtime/node_modules/pngjs')).PNG;
    session=await ort.InferenceSession.create(q.model,{executionProviders:['cpu'],intraOpNumThreads:4,interOpNumThreads:1,executionMode:'sequential',graphOptimizationLevel:'all'});
  }
  let calls=0;
  try{
    for(const c of q.cases){
      let im;
      if(!q.replay){
        im=PNG.sync.read(fs.readFileSync(path.join(q.output,c.key,'input.png')));
        if(im.width!==c.width||im.height!==c.height)throw Error('image shape');
      }
      for(const grid of [1,4]){
        const dir=path.join(q.output,c.key,`grid${grid}`),plan=tileMath.tiles(c.width,c.height,c.width,c.height,grid);
        const boxes=[],records=[];
        const previous=q.replay?JSON.parse(fs.readFileSync(path.join(dir,'result.json'),'utf8')):null;
        if(previous&&JSON.stringify(plan)!==JSON.stringify(previous.tiles.map(t=>t.window)))throw Error('tile metadata differs');
        if(!q.replay)fs.mkdirSync(dir,{recursive:false});
        for(const t of plan){
          if(q.replay){
            const b=zlib.gunzipSync(fs.readFileSync(path.join(dir,`tile-${t.index}.gz`)));
            boxes.push(...boxesFor(new Float32Array(b.buffer,b.byteOffset,b.length/4),previous.tiles[t.index].dims,t));
          }else{
            const start=performance.now(),input=math.prepare(grid===1?im:tileMath.crop(im,t)),prepared=performance.now();
            const outputs=await session.run({[session.inputNames[0]]:new ort.Tensor('float32',input.values,[1,3,640,640])});
            const inferred=performance.now(),scores=outputs[session.outputNames[0]];
            if(scores.type!=='float32')throw Error('unexpected output');
            boxes.push(...boxesFor(scores.data,scores.dims,t));
            const decoded=performance.now();
            write(path.join(dir,`tile-${t.index}.gz`),zlib.gzipSync(Buffer.from(scores.data.buffer,scores.data.byteOffset,scores.data.byteLength)));
            records.push({window:t,dims:scores.dims,preprocess_ms:prepared-start,inference_ms:inferred-prepared,decode_ms:decoded-inferred});
          }
          calls++;
        }
        const kept=math.nms(boxes);
        if(q.replay){if(JSON.stringify(kept)!==JSON.stringify(previous.boxes))throw Error('boxes replay differs');}
        else write(path.join(dir,'result.json'),JSON.stringify({grid,boxes:kept,tiles:records},null,2));
      }
      console.log(c.key,q.replay?'replayed':'inferred');
    }
  }finally{if(session)await session.release();}
  console.log(calls,'outputs',q.replay?'replayed':'saved');
}
if(require.main===module)main(JSON.parse(fs.readFileSync(0,'utf8'))).catch(e=>{console.error(e.stack);process.exitCode=1;});
