"""来源：本项目原创。按下一段最早停止期限保留可用采样帧，原地图证据不变。"""
from .detection_bridge import number
from .range_observation import RangeSamplingHistory


class ContinuationHistory(RangeSamplingHistory):
    """只清理不可能支持任何后续移动检查的帧，不扩大原 128 帧容量。

    物理检查的期限固定为 now + max_move_s + 0.5。时钟单调前进，因此
    valid_until < now + horizon 的帧不会在未来重新变得可用。
    已批准的主动阶段继续使用原 guard 中的来源/期限，原始输入仍完整归档。
    """
    def __init__(self, *, horizon_s=8.5, free_ttl_s=12., **kwargs):
        if (not number(horizon_s) or not number(free_ttl_s)
                or not 0 < horizon_s < free_ttl_s):
            raise ValueError('invalid fixed continuation horizon')
        super().__init__(**kwargs)
        self.horizon_s=horizon_s
        self.free_ttl_s=free_ttl_s
        self.retired=[]
        self.peak_frames=0

    def add(self, view, *, now_s):
        self._time(now_s)
        stop=now_s+self.horizon_s
        old=self._entries
        kept=[entry for entry in old if entry[0].captured_at_s+self.free_ttl_s+1e-9>=stop]
        retired=[dict(frame_id=v.frame_id,captured_at_s=v.captured_at_s,
                      valid_until_s=v.captured_at_s+self.free_ttl_s,
                      retired_at_s=now_s,earliest_next_stop_s=stop)
                 for v,_ in old if v.captured_at_s+self.free_ttl_s+1e-9<stop]
        self._entries=kept
        try:
            result=super().add(view,now_s=now_s)
        except Exception:
            # 不合格的新图像不能使已有账本发生半次更新。
            self._entries=old
            raise
        self.retired.extend(retired)
        self.peak_frames=max(self.peak_frames,len(self._entries))
        return result

    def check(self, start, end, *, now_s, budget, assumption=None):
        if (abs(budget.max_move_s+.5-self.horizon_s)>1e-9
                or abs(budget.free_ttl_s-self.free_ttl_s)>1e-9):
            raise ValueError('continuation history requires unchanged motion/time budget')
        return super().check(start,end,now_s=now_s,budget=budget,assumption=assumption)
