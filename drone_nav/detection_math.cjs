// 来源：本项目原创。按 YOLOX 官方输入/输出约定实现；通用插值、框解码和去重。
'use strict';
function prepare(image,size=640){
  if(!Number.isInteger(size)||size<=0||size>2048||![image.width,image.height].every(n=>Number.isInteger(n)&&n>0)||
     image.data.length!==image.width*image.height*4)throw Error('image shape');
  const ratio=Math.min(size/image.width,size/image.height);
  const width=Math.max(1,Math.floor(image.width*ratio)),height=Math.max(1,Math.floor(image.height*ratio));
  const values=new Float32Array(3*size*size).fill(114);
  // YOLOX 当前导出约定为 BGR、0..255，不套用 SegFormer 的 RGB/ImageNet 归一化。
  // 普通半像素双线性插值，不使用原分割预处理的抗混叠滤波；未声明与 cv2 逐位一致。
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const sx=Math.max(0,Math.min(image.width-1,(x+.5)*image.width/width-.5));
    const sy=Math.max(0,Math.min(image.height-1,(y+.5)*image.height/height-.5));
    const x0=Math.floor(sx),x1=Math.min(x0+1,image.width-1),y0=Math.floor(sy),y1=Math.min(y0+1,image.height-1);
    const fx=sx-x0,fy=sy-y0;
    for(let c=0;c<3;c++){
      const at=(xx,yy)=>image.data[(yy*image.width+xx)*4+(2-c)];
      values[c*size*size+y*size+x]=Math.round((at(x0,y0)*(1-fx)+at(x1,y0)*fx)*(1-fy)+
        (at(x0,y1)*(1-fx)+at(x1,y1)*fx)*fy);
    }
  }
  return {values,ratio,width,height};
}
function iou(a,b){
  const intersection=Math.max(0,Math.min(a.x2,b.x2)-Math.max(a.x1,b.x1))*Math.max(0,Math.min(a.y2,b.y2)-Math.max(a.y1,b.y1));
  const union=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-intersection;
  return union>0?intersection/union:0;
}
function nms(boxes,threshold=.45){
  if(!Number.isFinite(threshold)||threshold<0||threshold>1)throw Error('NMS threshold');
  const kept=[];
  for(const b of [...boxes].sort((a,b)=>b.score-a.score)){
    if(![b.x1,b.y1,b.x2,b.y2,b.score].every(Number.isFinite)||b.x2<=b.x1||b.y2<=b.y1||!['person','vehicle'].includes(b.group))throw Error('invalid box');
    if(!kept.some(a=>a.group===b.group&&iou(a,b)>threshold))kept.push(b);
  }
  return kept;
}
function decode(data,dims,prepared,tile,threshold=.3,size=640){
  const count=[8,16,32].reduce((s,stride)=>s+(size/stride)**2,0);
  if(dims.join()!==`1,${count},85`||data.length!==count*85||!data.every(Number.isFinite)||
     !Number.isFinite(threshold)||threshold<0||threshold>1)throw Error('detection output');
  const boxes=[];let index=0;
  for(const stride of [8,16,32])for(let gy=0;gy<size/stride;gy++)for(let gx=0;gx<size/stride;gx++,index++){
    const start=index*85;let cls=0;
    for(let c=1;c<80;c++)if(data[start+5+c]>data[start+5+cls])cls=c;
    const group=cls===0?'person':[2,5,7].includes(cls)?'vehicle':null;
    const score=data[start+4]*data[start+5+cls];
    if(!group||score<threshold)continue;
    const cx=(data[start]+gx)*stride,cy=(data[start+1]+gy)*stride;
    if(cx<0||cy<0||cx>=prepared.width||cy>=prepared.height)continue;
    const w=Math.exp(data[start+2])*stride,h=Math.exp(data[start+3])*stride;
    const clamp=(v,hi)=>Math.max(0,Math.min(hi,v));
    const box={x1:tile.x0+clamp((cx-w/2)/prepared.ratio,tile.width),
      y1:tile.y0+clamp((cy-h/2)/prepared.ratio,tile.height),
      x2:tile.x0+clamp((cx+w/2)/prepared.ratio,tile.width),
      y2:tile.y0+clamp((cy+h/2)/prepared.ratio,tile.height),score,group,coco_class:cls,tile:tile.index};
    if(box.x2>box.x1&&box.y2>box.y1)boxes.push(box);
  }
  return boxes;
}
module.exports={prepare,iou,nms,decode};
