"""来源：本项目原创。漏检流水线支持证据、跨块边界统计与匹配集合变化。"""
from .object_metrics import iou, size_name

REASONS = ('matching_competition','nms_support_removed','core_support_removed','low_score_support',
           'localization_support','other_group_overlap','no_qualifying_support')


def geometry(b):
    return [b[k] for k in ('x1','y1','x2','y2')]


def crosses_seam(box, width, height, grid):
    # 与 tileMath.tiles 的原图像素负责区边界完全一致，不使用上下文裁剪边界。
    return any(box[0] < (k*width//grid) < box[2] or box[1] < (k*height//grid) < box[3] for k in range(1,grid))


def best_support(target, boxes, same=True, low=.5, high=1.0):
    choices=[]
    for index,b in enumerate(boxes):
        if (b['group']==target['group']) != same:
            continue
        overlap=iou(target['box'],geometry(b))
        if low <= overlap <= high:
            choices.append((overlap,b['score'],-index,b))
    if not choices:
        return None
    overlap,_,_,b=max(choices,key=lambda x:x[:3])
    return dict(candidate=b['audit_id'],iou=overlap,score=b['score'],tile=b['tile'],owned=b['owned'])


def diagnose(target, trace):
    candidates=trace['candidates']
    high=[b for b in candidates if b['score']>=.3]
    core=[b for b in high if b['owned']]
    checks=[('matching_competition',trace['variants']['baseline'],True,.5,1),
            ('nms_support_removed',core,True,.5,1),('core_support_removed',high,True,.5,1),
            ('low_score_support',[b for b in candidates if b['score']<.3],True,.5,1),
            ('localization_support',high,True,.1,.5),('other_group_overlap',high,False,.5,1)]
    for reason,boxes,same,low,upper in checks:
        evidence=best_support(target,boxes,same,low,upper)
        if evidence:
            if reason=='nms_support_removed':
                evidence['suppression']=next(s for s in trace['suppressions']['baseline'] if s['removed']==evidence['candidate'])
            return dict(annotation=target['id'],group=target['group'],reason=reason,evidence=evidence,
                        size=size_name(target['box']),occlusion=target['occlusion'],truncation=target['truncation'])
    return dict(annotation=target['id'],group=target['group'],reason='no_qualifying_support',evidence=None,
                size=size_name(target['box']),occlusion=target['occlusion'],truncation=target['truncation'])


def changes(before, after):
    old={m['annotation'] for m in before['matches']}
    new={m['annotation'] for m in after['matches']}
    return dict(recovered=sorted(new-old),lost=sorted(old-new),retained=sorted(old&new))
