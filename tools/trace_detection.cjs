// 来源：本项目原创。从已存分数重放后处理，不调用模型；旧代码不改动。
'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const math=require('../drone_nav/detection_math.cjs');
const tiles=require('../drone_nav/segmentation_tiles.cjs');
function owns(b,t){const x=(b.x1+b.x2)/2,y=(b.y1+b.y2)/2;return x>=t.ex0&&x<t.ex0+t.eval_width&&y>=t.ey0&&y<t.ey0+t.eval_height;}
function plain(b){const {audit_id,owned,...box}=b;return box;}
function suppressions(candidates,kept){
  const ranks=new Map(candidates.map((b,i)=>[b.audit_id,i])),keepIds=new Set(kept.map(b=>b.audit_id));
  return candidates.filter(b=>!keepIds.has(b.audit_id)).map(b=>{
    const by=kept.find(k=>k.group===b.group&&(k.score>b.score||(k.score===b.score&&ranks.get(k.audit_id)<ranks.get(b.audit_id)))&&math.iou(k,b)>.45);
    if(!by)throw Error('missing suppression witness');
    return {removed:b.audit_id,kept:by.audit_id,iou:math.iou(by,b)};
  });
}
function variants(candidates){
  const high=candidates.filter(b=>b.score>=.3),core=high.filter(b=>b.owned);
  const baseline=math.nms(core),noCore=math.nms(high);
  return {variants:{baseline,no_core:noCore,no_nms:core,no_core_no_nms:high},
    suppressions:{baseline:suppressions(core,baseline),no_core:suppressions(high,noCore)}};
}
function main(q){
  const cases=JSON.parse(fs.readFileSync(path.join(q.input,'data.json'),'utf8')).cases;
  let count=0;
  for(const c of cases)for(const grid of [1,4]){
    const dir=path.join(q.input,c.key,`grid${grid}`),meta=JSON.parse(fs.readFileSync(path.join(dir,'result.json'),'utf8'));
    const plan=tiles.tiles(c.width,c.height,c.width,c.height,grid),candidates=[];
    if(JSON.stringify(plan)!==JSON.stringify(meta.tiles.map(t=>t.window)))throw Error('tile plan differs');
    for(const t of plan){
      const raw=zlib.gunzipSync(fs.readFileSync(path.join(dir,`tile-${t.index}.gz`))),ratio=Math.min(640/t.width,640/t.height);
      const decoded=math.decode(new Float32Array(raw.buffer,raw.byteOffset,raw.length/4),meta.tiles[t.index].dims,
        {ratio,width:Math.floor(t.width*ratio),height:Math.floor(t.height*ratio)},t,.05);
      decoded.forEach((b,i)=>candidates.push({...b,audit_id:`${t.index}:${i}`,owned:owns(b,t)}));count++;
    }
    const trace=variants(candidates);
    if(JSON.stringify(trace.variants.baseline.map(plain))!==JSON.stringify(meta.boxes))throw Error('baseline boxes differ');
    if(grid===1&&JSON.stringify(trace.variants.baseline)!==JSON.stringify(trace.variants.no_core))throw Error('whole image differs');
    const out=path.join(q.output,`${c.key}-grid${grid}.json`);
    const record=JSON.stringify({key:c.key,grid,candidates,...trace});
    if(q.verify){if(fs.readFileSync(out,'utf8')!==record)throw Error('trace replay differs');}
    else fs.writeFileSync(out,record,{flag:'wx'});
  }
  console.log(count,'outputs decoded; 24 baselines identical; 12 whole-image no-core controls identical');
}
module.exports={owns,plain,suppressions,variants};
if(require.main===module)main(JSON.parse(fs.readFileSync(0,'utf8')));
