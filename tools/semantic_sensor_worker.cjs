// 来源：本项目原创。常驻本地 ONNX 会话；每帧保存原始输出并复用既有解码。
'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const readline=require('node:readline');
const {performance}=require('node:perf_hooks');
const {decode}=require('../drone_nav/tinyformer_math.cjs');
const send=x=>process.stdout.write(JSON.stringify(x)+'\n');
const floats=p=>{const b=zlib.gunzipSync(fs.readFileSync(p));if(b.length%4)throw Error('unaligned tensor');return new Float32Array(b.buffer,b.byteOffset,b.length/4);};
const write=(p,v)=>fs.writeFileSync(p,v,{flag:'wx'});
async function main(){
  let session,ort,init;
  try{
    for await(const line of readline.createInterface({input:process.stdin,crlfDelay:Infinity})){
      const q=JSON.parse(line);
      if(!init){
        init=q;
        if(!q.replay){
          ort=require(path.join(q.runtime,'runtime/node_modules/onnxruntime-node'));
          session=await ort.InferenceSession.create(q.model,{executionProviders:['cpu'],intraOpNumThreads:4,interOpNumThreads:1,executionMode:'sequential',graphOptimizationLevel:'all'});
          if(session.inputNames.join()!=='images'||session.outputNames.join()!=='pred_logits,pred_boxes')throw Error('model interface');
        }
        send({ready:true});continue;
      }
      const dir=q.directory,t=q.window;
      if(init.replay){
        const old=JSON.parse(fs.readFileSync(path.join(dir,'result.json'),'utf8'));
        const result=decode(floats(path.join(dir,'logits.gz')),old.logits_dims,
          floats(path.join(dir,'boxes.gz')),old.boxes_dims,t);
        if(JSON.stringify(result.boxes)!==JSON.stringify(old.boxes)||result.degenerate!==old.degenerate)throw Error('redecoded output differs');
        send({replayed:true,boxes:result.boxes});continue;
      }
      const input=floats(path.join(dir,'input.gz'));
      if(input.length!==3*640*640||!input.every(Number.isFinite))throw Error('input tensor');
      const start=performance.now();
      const out=await session.run({images:new ort.Tensor('float32',input,[1,3,640,640])});
      const inference_ms=performance.now()-start,l=out.pred_logits,b=out.pred_boxes;
      if(l.type!=='float32'||b.type!=='float32')throw Error('output dtype');
      const decoded=decode(l.data,l.dims,b.data,b.dims,t);
      for(const [name,data]of [['logits',l.data],['boxes',b.data]])
        write(path.join(dir,name+'.gz'),zlib.gzipSync(Buffer.from(data.buffer,data.byteOffset,data.byteLength)));
      const result={...decoded,logits_dims:l.dims,boxes_dims:b.dims,inference_ms,window:t};
      write(path.join(dir,'result.json'),JSON.stringify(result));send(result);
    }
  }finally{if(session)await session.release();}
}
main().catch(e=>{send({error:e.message});process.exitCode=1;});
