// 来源：本项目原创。按发布者公开接口解码 TinyFormer，不执行模型仓库代码。
'use strict';
function decode(logits, ldims, coords, bdims, tile, threshold=.3){
  if(ldims.join()!=='1,300,10'||bdims.join()!=='1,300,4'||logits.length!==3000||coords.length!==1200||
     !logits.every(Number.isFinite)||!coords.every(Number.isFinite)||
     !Number.isFinite(threshold)||threshold<0||threshold>1)throw Error('TinyFormer output shape or values');
  if(![tile.width,tile.height].every(n=>Number.isInteger(n)&&n>0)||
     ![tile.x0,tile.y0,tile.index].every(n=>Number.isInteger(n)&&n>=0))throw Error('tile geometry');
  // 每个 query 可以有多个类别候选；先在全部十类中选前 300，再过滤业务类别。
  const ranked=Array.from(logits,(v,i)=>({score:Math.fround(v>=0?1/(1+Math.exp(-v)):Math.exp(v)/(1+Math.exp(v))),i}));
  ranked.sort((a,b)=>b.score-a.score||a.i-b.i);
  const boxes=[];let degenerate=0;
  for(const {score,i} of ranked.slice(0,300)){
    const cls=i%10,query=Math.floor(i/10),group=[0,1].includes(cls)?'person':[3,4,5,8].includes(cls)?'vehicle':null;
    if(!group||score<threshold)continue;
    const [cx,cy,w,h]=coords.slice(query*4,query*4+4);
    if(w<0||h<0)throw Error('negative predicted size');
    const clip=(v,n)=>Math.max(0,Math.min(n,v));
    const b={x1:tile.x0+clip((cx-w/2)*tile.width,tile.width),y1:tile.y0+clip((cy-h/2)*tile.height,tile.height),
      x2:tile.x0+clip((cx+w/2)*tile.width,tile.width),y2:tile.y0+clip((cy+h/2)*tile.height,tile.height),
      score,group,visdrone_class:cls+1,query,tile:tile.index};
    if(b.x2<=b.x1||b.y2<=b.y1){degenerate++;continue;}
    boxes.push(b);
  }
  return {boxes,degenerate};
}
function owns(b,t){const x=(b.x1+b.x2)/2,y=(b.y1+b.y2)/2;return x>=t.ex0&&x<t.ex0+t.eval_width&&y>=t.ey0&&y<t.ey0+t.eval_height;}
module.exports={decode,owns};
