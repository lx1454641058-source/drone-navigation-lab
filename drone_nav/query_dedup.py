"""来源：本项目原创。按模型位置候选合并业务类别，保留原始索引证据。"""
import math


def merge_queries(boxes):
    geometry={}
    for b in boxes:
        if any(type(b.get(k)) is not int or b[k]<0 for k in ('tile','query')) or b['query']>=300:
            raise ValueError('invalid query identity')
        coords=tuple(b.get(k) for k in ('x1','y1','x2','y2'))
        if (b.get('group') not in ('person','vehicle') or
            any(type(v) not in (int,float) or not math.isfinite(v) for v in (*coords,b.get('score'))) or
            not 0<=b['score']<=1 or coords[2]<=coords[0] or coords[3]<=coords[1]):
            raise ValueError('invalid prediction')
        key=(b['tile'],b['query'])
        if key in geometry and geometry[key]!=coords:
            raise ValueError('one query has inconsistent geometry')
        geometry[key]=coords
    selected={};kept=[];removed=[]
    # Python 排序稳定；同分保留输入先后顺序，与既有解码规则一致。
    for i in sorted(range(len(boxes)),key=lambda i:-boxes[i]['score']):
        b=boxes[i];key=(b['tile'],b['query'],b['group'])
        if key in selected:
            removed.append(dict(prediction=i,kept_prediction=selected[key],reason='same_query_and_business_group'))
        else:
            selected[key]=i;kept.append(i)
    return dict(boxes=[boxes[i] for i in kept],kept_predictions=kept,removed=removed)
