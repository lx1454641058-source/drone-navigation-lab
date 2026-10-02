// 来源：本项目原创。两个开发帧的候选模型探针，复用已验证的采样与预处理。
'use strict';
const fs = require('node:fs'), path = require('node:path'), zlib = require('node:zlib');
const {performance} = require('node:perf_hooks');
const old = require('../drone_nav/segmentation_worker.cjs');
const math = require('../drone_nav/segmentation_tiles.cjs');
const put = (file, data) => fs.writeFileSync(file, data, {flag:'wx'});
async function main(q) {
  if (q.cases.length !== 2 || q.cases.map(c=>c.key).join() !== 'seq1_000000,seq2_000000')
    throw Error('development cases only');
  if (q.replay) {
    for (const c of q.cases) for (const grid of [1,4]) {
      const dir=path.join(q.output,c.key,`grid${grid}`);
      const m=JSON.parse(fs.readFileSync(path.join(dir,'worker.json'),'utf8'));
      const plan=math.tiles(m.source_width,m.source_height,m.width,m.height,grid);
      if (JSON.stringify(plan)!==JSON.stringify(m.records.map(t=>t.window))) throw Error('tile plan differs');
      const pred=new Uint8Array(m.width*m.height), coverage=new Uint8Array(pred.length);
      for (const t of m.records) {
        const b=zlib.gunzipSync(fs.readFileSync(path.join(dir,`tile-${t.window.index}.gz`)));
        const scores=new Float32Array(b.buffer,b.byteOffset,b.length/4);
        if (t.dims.join()!=='1,19,128,128') throw Error('output shape differs');
        math.stitch(pred,m.width,m.height,t.window,math.classifyTile(scores,t.dims,t.window,
          m.source_width,m.source_height,m.width,m.height),coverage);
      }
      if (!coverage.every(v=>v===1) || !Buffer.from(pred).equals(fs.readFileSync(path.join(dir,'city.u8'))))
        throw Error('replay differs');
    }
    console.log('replay passed: 4 predictions, 34 tiles'); return;
  }
  const ort=require(path.join(q.root,'runtime/node_modules/onnxruntime-node'));
  const {PNG}=require(path.join(q.root,'runtime/node_modules/pngjs'));
  const config=JSON.parse(fs.readFileSync(path.join(q.model,'preprocessor_config.json'),'utf8'));
  const session=await ort.InferenceSession.create(path.join(q.model,'model.onnx'),{
    executionProviders:['cpu'],intraOpNumThreads:4,interOpNumThreads:1,executionMode:'sequential',graphOptimizationLevel:'all'});
  for (const c of q.cases) {
    const t0=performance.now();
    const im=PNG.sync.read(fs.readFileSync(path.join(q.root,c.image)));
    for(let i=3;i<im.data.length;i+=4) if(im.data[i]!==255) throw Error('transparent input');
    const decode=performance.now()-t0, width=960, height=Math.round(im.height*width/im.width);
    for (const grid of [1,4]) {
      const dir=path.join(q.output,c.key,`grid${grid}`);fs.mkdirSync(dir,{recursive:true});
      const pred=new Uint8Array(width*height), coverage=new Uint8Array(pred.length),records=[];
      for(const tile of math.tiles(im.width,im.height,width,height,grid)) {
        const start=performance.now();
        const rgb=old.resizeRGB(grid===1?im:math.crop(im,tile),512,512);
        const input=new ort.Tensor('float32',old.normalize(rgb,config),[1,3,512,512]);
        const prepared=performance.now();
        const result=await session.run({[session.inputNames[0]]:input});
        const scores=result[session.outputNames[0]],computed=performance.now();
        if(scores.type!=='float32'||scores.dims.join()!=='1,19,128,128')throw Error('unexpected output');
        math.stitch(pred,width,height,tile,math.classifyTile(scores.data,scores.dims,tile,im.width,im.height,width,height),coverage);
        const done=performance.now();
        put(path.join(dir,`tile-${tile.index}.gz`),zlib.gzipSync(Buffer.from(scores.data.buffer,scores.data.byteOffset,scores.data.byteLength)));
        records.push({window:tile,dims:scores.dims,preprocess_ms:prepared-start,inference_ms:computed-prepared,postprocess_ms:done-computed});
      }
      if(!coverage.every(v=>v===1))throw Error('coverage');
      const core_ms=decode+records.reduce((s,t)=>s+t.preprocess_ms+t.inference_ms+t.postprocess_ms,0);
      put(path.join(dir,'city.u8'),pred);
      put(path.join(dir,'worker.json'),JSON.stringify({source_width:im.width,source_height:im.height,width,height,decode_ms:decode,core_ms,records}));
      console.log(JSON.stringify({key:c.key,grid,core_ms}));
    }
  }
  await session.release();
}
if(require.main===module)main(JSON.parse(fs.readFileSync(0,'utf8'))).catch(e=>{console.error(e.stack);process.exitCode=1;});
