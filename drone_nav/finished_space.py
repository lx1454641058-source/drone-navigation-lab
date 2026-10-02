"""来源：本项目原创。空间消费结束后的时效、状态与控制截止复查。"""
from time import perf_counter

from .detection_bridge import number
from .following_space import consume_following_space
from .stop_forecast import FrozenForecast


def _monitor_identity(monitor):
    return (monitor.reason, monitor.request.sha256,
            None if monitor.packet is None else monitor.packet.sha256, monitor.last_index)


def consume_finished_space(receipt, monitor, current, reference, *, capture_wall_s,
                           cycle_deadline_wall_s, clock_id, checkpoint_reader,
                           clock_reader=perf_counter, wall_clock_id='host-perf-counter',
                           deadline_missed=False):
    """Consume then read the actual monotonic clock after validation work.

    The checkpoint reader samples physics/controller identity after consumption.
    No movement is authorized. A later actuator handoff still needs its own
    deadline check; this function only closes the known consume-start gap.
    """
    if not number(capture_wall_s) or not number(cycle_deadline_wall_s):
        raise ValueError('finite same-host capture and control deadline required')
    sealed = receipt.record()
    query = FrozenForecast(**sealed['query']).record()
    identity = _monitor_identity(monitor)
    same_wall_clock = wall_clock_id == sealed['wall_clock_id']
    started = clock_reader()
    if not number(started) or (same_wall_clock and started < capture_wall_s):
        raise ValueError('invalid consumption start or future capture')
    provisional = consume_following_space(receipt, monitor, current, reference,
        now_s=current['time_s'], clock_id=clock_id, wall_clock_id=wall_clock_id,
        wall_age_s=started-capture_wall_s if same_wall_clock else 0.,
        deadline_missed=deadline_missed or (same_wall_clock and started > cycle_deadline_wall_s))
    after = checkpoint_reader()
    failures = []
    if after != current or _monitor_identity(monitor) != identity:
        failures.append('STATE_OR_MONITOR_CHANGED_DURING_CONSUMPTION')
    # This read follows receipt parsing, geometry-result consumption and the
    # second full-state checkpoint. Never reuse the timestamp from entry.
    finished = clock_reader()
    if not number(finished):
        raise ValueError('finite consumption completion time required')
    if finished < started:
        failures.append('WALL_CLOCK_REVERSED_DURING_CONSUMPTION')
    if same_wall_clock:
        if finished > cycle_deadline_wall_s:
            failures.append('CONSUMPTION_FINISHED_AFTER_CONTROL_DEADLINE')
        if finished-capture_wall_s > sealed['wall_age_limit_s']+1e-9:
            failures.append('DEPTH_EXPIRED_DURING_CONSUMPTION')
    # A changed state remains rejected even if it has a later matching prediction.
    # Only compare times after proving clock identity, as in the frozen adapter.
    if clock_id == query['clock_id'] == query['depth_clock_id'] == after.get('clock') == monitor.record['binding']['clock']:
        if after['time_s'] > query['depth_valid_until_s']+1e-9:
            failures.append('DEPTH_EXPIRED_AT_FINAL_CHECKPOINT')
    return dict(provisional=provisional, started_wall_s=started, finished_wall_s=finished,
        final_checkpoint=after, capture_wall_s=capture_wall_s, cycle_deadline_wall_s=cycle_deadline_wall_s,
        final_wall_age_s=finished-capture_wall_s if same_wall_clock else None, final_failures=failures,
        diagnostic_current=provisional['diagnostic_current'] and not failures,
        reasons=list(dict.fromkeys(provisional['reasons']+failures)), status='HOLD',
        flight_authorized=False, selected_for_execution=False, free_volume_proven=False,
        navigation_map_update_allowed=False)
