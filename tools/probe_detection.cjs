// 来源：本项目原创。独立开发探针与原始模型输出重放；不发飞行指令。
'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const {performance}=require('node:perf_hooks');
const math=require('../drone_nav/detection_math.cjs'),tiles=require('../drone_nav/segmentation_tiles.cjs');
const write=(p,v)=>fs.writeFileSync(p,v,{flag:'wx'});
function coreBoxes(data,dims,t,sw,sh){
  const ratio=Math.min(640/t.width,640/t.height);
  const prepared={ratio,width:Math.floor(t.width*ratio),height:Math.floor(t.height*ratio)};
  return math.decode(data,dims,prepared,t).filter(b=>{
    const x=(b.x1+b.x2)/2*960/sw,y=(b.y1+b.y2)/2*540/sh;
    return x>=t.ex0&&x<t.ex0+t.eval_width&&y>=t.ey0&&y<t.ey0+t.eval_height;
  });
}
async function main(q){
  if(q.cases.map(c=>c.key).join()!=='seq1_000000,seq2_000000')throw Error('development cases only');
  if(q.replay){
    for(const c of q.cases)for(const grid of [1,4]){
      const dir=path.join(q.output,c.key,`grid${grid}`);
      const meta=JSON.parse(fs.readFileSync(path.join(dir,'result.json'),'utf8'));
      const plan=tiles.tiles(meta.width,meta.height,960,540,grid);
      if(JSON.stringify(plan)!==JSON.stringify(meta.tiles.map(t=>t.window)))throw Error('tile metadata');
      const boxes=[];
      for(const r of meta.tiles){
        const b=zlib.gunzipSync(fs.readFileSync(path.join(dir,`tile-${r.window.index}.gz`)));
        boxes.push(...coreBoxes(new Float32Array(b.buffer,b.byteOffset,b.length/4),r.dims,r.window,meta.width,meta.height));
      }
      if(JSON.stringify(math.nms(boxes))!==JSON.stringify(meta.boxes))throw Error('boxes replay differs');
    }
    console.log('34 outputs replayed; 4 box lists identical');return;
  }
  const ort=require(path.join(q.root,'runtime/node_modules/onnxruntime-node'));
  const {PNG}=require(path.join(q.root,'runtime/node_modules/pngjs'));
  const session=await ort.InferenceSession.create(q.model,{executionProviders:['cpu'],intraOpNumThreads:4,
    interOpNumThreads:1,executionMode:'sequential',graphOptimizationLevel:'all'});
  for(const c of q.cases){
    const im=PNG.sync.read(fs.readFileSync(path.join(q.root,c.image)));
    if(im.width!==3840||im.height!==2160)throw Error('unexpected development dimensions');
    for(let i=3;i<im.data.length;i+=4)if(im.data[i]!==255)throw Error('transparent source');
    for(const grid of [1,4]){
      const dir=path.join(q.output,c.key,`grid${grid}`);fs.mkdirSync(dir,{recursive:true});
      const records=[],boxes=[];
      for(const t of tiles.tiles(im.width,im.height,960,540,grid)){
        const start=performance.now();
        const input=math.prepare(grid===1?im:tiles.crop(im,t));
        const prepared=performance.now();
        const result=await session.run({[session.inputNames[0]]:new ort.Tensor('float32',input.values,[1,3,640,640])});
        const inferred=performance.now(),scores=result[session.outputNames[0]];
        if(scores.type!=='float32')throw Error('unexpected tensor type');
        boxes.push(...coreBoxes(scores.data,scores.dims,t,im.width,im.height));
        const decoded=performance.now();
        write(path.join(dir,`tile-${t.index}.gz`),zlib.gzipSync(Buffer.from(scores.data.buffer,scores.data.byteOffset,scores.data.byteLength)));
        records.push({window:t,dims:scores.dims,preprocess_ms:prepared-start,inference_ms:inferred-prepared,decode_ms:decoded-inferred});
      }
      const kept=math.nms(boxes);
      write(path.join(dir,'result.json'),JSON.stringify({width:im.width,height:im.height,grid,boxes:kept,tiles:records},null,2));
      console.log(c.key,grid,kept.length,'boxes');
    }
  }
  await session.release();
}
if(require.main===module)main(JSON.parse(fs.readFileSync(0,'utf8'))).catch(e=>{console.error(e.stack);process.exitCode=1;});
