// 来源：本项目原创。可配置 CPU 线程与张量存储，复用原 TinyFormer 解码。
'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const readline=require('node:readline');
const {performance}=require('node:perf_hooks');
const {decode}=require('../drone_nav/tinyformer_math.cjs');
const send=x=>process.stdout.write(JSON.stringify(x)+'\n');
const floats=b=>{if(b.length%4)throw Error('unaligned float32');return new Float32Array(b.buffer,b.byteOffset,b.length/4);};
const unzip=p=>zlib.gunzipSync(fs.readFileSync(p));
const write=(p,v)=>fs.writeFileSync(p,v,{flag:'wx'});
async function main(){
  let session,ort,init;
  try{
    for await(const line of readline.createInterface({input:process.stdin,crlfDelay:Infinity})){
      const q=JSON.parse(line);
      if(!init){
        if(![4,8,16].includes(q.threads)||!['gzip','raw'].includes(q.input_mode))throw Error('unsupported runtime options');
        init=q;
        if(!q.replay){
          ort=require(path.join(q.runtime,'runtime/node_modules/onnxruntime-node'));
          session=await ort.InferenceSession.create(q.model,{executionProviders:['cpu'],intraOpNumThreads:q.threads,interOpNumThreads:1,executionMode:'sequential',graphOptimizationLevel:'all'});
          if(session.inputNames.join()!=='images'||session.outputNames.join()!=='pred_logits,pred_boxes')throw Error('model interface');
        }
        send({ready:true,threads:q.threads,input_mode:q.input_mode});continue;
      }
      const dir=q.directory,t=q.window;
      if(init.replay){
        const old=JSON.parse(fs.readFileSync(path.join(dir,'result.json'),'utf8'));
        const decoded=decode(floats(unzip(path.join(dir,'logits.gz'))),old.logits_dims,
          floats(unzip(path.join(dir,'boxes.gz'))),old.boxes_dims,t);
        if(JSON.stringify(decoded.boxes)!==JSON.stringify(old.boxes)||decoded.degenerate!==old.degenerate)throw Error('decoded output differs');
        send({replayed:true,boxes:decoded.boxes});continue;
      }
      const started=performance.now();
      const input=floats(init.input_mode==='gzip'?unzip(path.join(dir,'input.gz')):fs.readFileSync(path.join(dir,'input.f32')));
      const loaded=performance.now();
      if(input.length!==3*640*640||!input.every(Number.isFinite))throw Error('invalid input tensor');
      const validated=performance.now();
      const out=await session.run({images:new ort.Tensor('float32',input,[1,3,640,640])});
      const inferred=performance.now(),l=out.pred_logits,b=out.pred_boxes;
      if(l.type!=='float32'||b.type!=='float32')throw Error('output dtype');
      const decoded=decode(l.data,l.dims,b.data,b.dims,t),decodedAt=performance.now();
      for(const [name,data]of [['logits',l.data],['boxes',b.data]])
        write(path.join(dir,name+'.gz'),zlib.gzipSync(Buffer.from(data.buffer,data.byteOffset,data.byteLength)));
      const result={...decoded,logits_dims:l.dims,boxes_dims:b.dims,inference_ms:inferred-validated,window:t,
        worker_stages_ms:{read_input:loaded-started,validate_input:validated-loaded,
          inference:inferred-validated,decode:decodedAt-inferred,archive_outputs:performance.now()-decodedAt}};
      write(path.join(dir,'result.json'),JSON.stringify(result));send(result);
    }
  }finally{if(session)await session.release();}
}
main().catch(e=>{send({error:e.message});process.exitCode=1;});
