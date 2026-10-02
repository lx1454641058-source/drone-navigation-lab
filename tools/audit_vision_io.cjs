// 来源：本项目原创。输入与运行优化对照，只处理固定的两个开发帧。
'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),zlib=require('node:zlib');
const old=require('../drone_nav/segmentation_worker.cjs');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const put=(p,b)=>fs.writeFileSync(p,b,{flag:'wx'});
async function main(q){
  const {PNG}=require(path.join(q.root,'runtime/node_modules/pngjs'));
  const ort=require(path.join(q.root,'runtime/node_modules/onnxruntime-node'));
  const pre=JSON.parse(fs.readFileSync(path.join(q.root,'model/preprocessor_config.json'),'utf8'));
  const inputs={};
  for(const seq of [1,2]){
    const key=`seq${seq}_000000`,dir=path.join(q.output,key);
    const im=PNG.sync.read(fs.readFileSync(path.join(q.root,`data/images/train/${key}.png`)));
    const lab=PNG.sync.read(fs.readFileSync(path.join(q.root,`data/masks/train/${key}.png`)));
    const rgb=old.resizeRGB(im,512,512),tensor=old.normalize(rgb,pre);
    put(path.join(dir,'js.rgb'),rgb);
    put(path.join(dir,'js.f32'),Buffer.from(tensor.buffer));
    put(path.join(dir,'js.label'),old.labelGrid(lab,960,540));
    put(path.join(dir,'decoded.json'),JSON.stringify({image_sha:sha(im.data),label_sha:sha(lab.data),width:im.width,height:im.height}));
    const pil=fs.readFileSync(path.join(dir,'pillow.f32'));
    inputs[key]={js:tensor,pillow:new Float32Array(pil.buffer,pil.byteOffset,pil.length/4)};
  }
  for(const model of ['ade','city'])for(const optimization of ['all','disabled']){
    const file=model==='ade'?path.join(q.root,'model/model.onnx'):path.join(q.city,'model.onnx');
    const session=await ort.InferenceSession.create(file,{executionProviders:['cpu'],intraOpNumThreads:4,
      interOpNumThreads:1,executionMode:'sequential',graphOptimizationLevel:optimization});
    for(const key of Object.keys(inputs))for(const input of optimization==='all'?['js']:['js','pillow']){
      const outputs=await session.run({[session.inputNames[0]]:new ort.Tensor('float32',inputs[key][input],[1,3,512,512])});
      const scores=outputs[session.outputNames[0]];
      if(scores.dims.join()!==`1,${model==='ade'?150:19},128,128`)throw Error('unexpected shape');
      const prefix=path.join(q.output,key,`${model}-${optimization}-${input}`);
      put(prefix+'.u8',old.argmaxResize(scores.data,scores.dims,960,540));
      put(prefix+'.f32.gz',zlib.gzipSync(Buffer.from(scores.data.buffer,scores.data.byteOffset,scores.data.byteLength)));
      console.log(model,optimization,input,key);
    }
    await session.release();
  }
}
if(require.main===module)main(JSON.parse(fs.readFileSync(0,'utf8'))).catch(e=>{console.error(e.stack);process.exitCode=1;});
