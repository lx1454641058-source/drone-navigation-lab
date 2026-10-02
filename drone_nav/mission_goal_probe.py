"""来源：本项目原创。观察前沿是中间点，只有配送任务目标可以免去继续取景。"""


class MissionGoalProbeMixin:
    def move(self,target_cell):
        if not self.spec.get('mission_goal_probe',False):return super().move(target_cell)
        missing=object()
        previous=getattr(self,'_navigation_aim',missing)
        # 原主动观察包装器用此字段决定是否为最后一段。扫描方向仍由
        # 原观察策略选择；这里只给动作完成判断提供明确的任务目标。
        self._navigation_aim=tuple(self.spec['goal'])
        try:return super().move(target_cell)
        finally:
            if previous is missing:del self._navigation_aim
            else:self._navigation_aim=previous
