"""来源：本项目原创。慢空间查询交付后，按同预测的剩余子范围复查诊断依据。"""
from dataclasses import asdict

from .async_stop import ForecastMonitor
from .detection_bridge import number
from .forecast_space import SimulationReference, query_stop_space
from .stop_forecast import FrozenForecast, digest


def prepare_following_space(monitor, current, inspector, reference, *, now_s, clock_id):
    """Worker-only: seal the checked superset and short-lived immutable state proofs.

    A sampled superset is not free-space evidence. Its original negative/unknown
    findings remain conservative restrictions even if they lie outside a suffix.
    """
    query = query_stop_space(monitor, current, inspector, reference, now_s=now_s, clock_id=clock_id)
    record = query.record()
    proofs = []
    if record['geometry'] is not None:
        for point in monitor.packet.record()['checkpoints']:
            if (current['index'] <= point['index'] < 4000
                    and point['time_s'] <= inspector.valid_until_s + 1e-9):
                remaining = point['remaining']
                proofs.append(dict(index=point['index'], time_s=point['time_s'],
                    checkpoint_sha256=digest({k: v for k, v in point.items() if k != 'remaining'}),
                    lower=[v-record['relative_error_padding_m'] for v in remaining['lower_m']],
                    upper=[v+record['relative_error_padding_m'] for v in remaining['upper_m']]))
    return FrozenForecast.create(dict(schema='following-stop-space-v1', query=asdict(query),
        proofs=proofs, depth_captured_at_s=inspector.captured_at_s,
        wall_clock_id='host-perf-counter',
        wall_age_limit_s=inspector.valid_until_s-inspector.captured_at_s,
        flight_authorized=False))


def consume_following_space(receipt, monitor, current, reference, *, now_s, clock_id,
                            wall_age_s, wall_clock_id='host-perf-counter', deadline_missed=False):
    """Fast diagnostic check using a sealed result, not a mutable depth buffer.

    Wall age is measured by the caller from the actual capture event on the same
    host monotonic clock. It is never subtracted from the model timestamp.
    The separate checks also reject a result during a simulation-time catch-up.
    """
    if type(receipt) is not FrozenForecast or type(monitor) is not ForecastMonitor or type(reference) is not SimulationReference:
        raise TypeError('immutable receipt, stop monitor and explicit reference required')
    if not number(now_s) or now_s < 0 or not number(wall_age_s) or wall_age_s < 0:
        raise ValueError('finite nonnegative model time and separately measured wall age required')
    sealed = receipt.record()
    if sealed['schema'] != 'following-stop-space-v1' or sealed['flight_authorized'] is not False:
        raise ValueError('invalid spatial receipt schema or authorization claim')
    query = FrozenForecast(**sealed['query']).record()
    if query['schema'] != 'stop-space-query-v1':
        raise ValueError('invalid spatial query schema')
    failures = []
    packet_sha = None if monitor.packet is None else monitor.packet.sha256
    if monitor.reason != 'MATCHED':
        failures.append('STOP_PREDICTION_NOT_MATCHED:'+monitor.reason)
    if (monitor.request.sha256 != query['request_sha256'] or packet_sha != query['prediction_sha256']):
        failures.append('PREDICTION_IDENTITY_CHANGED')
    if asdict(reference) != query['reference']:
        failures.append('REFERENCE_CHANGED')
    same_clock = clock_id == query['clock_id'] == query['depth_clock_id'] == current.get('clock') == monitor.record['binding']['clock'] == reference.clock_id
    if not same_clock:
        failures.append('COMPLETION_CLOCK_MISMATCH')
    if wall_clock_id != sealed['wall_clock_id']:
        failures.append('WALL_CLOCK_MISMATCH')
    elif wall_age_s > sealed['wall_age_limit_s'] + 1e-9:
        failures.append('DEPTH_EXPIRED_ON_WALL_CLOCK')
    if same_clock:
        if now_s < query['checked_at_s'] or now_s != current.get('time_s'):
            failures.append('CURRENT_TIME_MISMATCH')
        if now_s > query['depth_valid_until_s'] + 1e-9:
            failures.append('DEPTH_EXPIRED_ON_MODEL_CLOCK')
        if query['horizon_end_s'] is not None and now_s >= query['horizon_end_s']:
            failures.append('STOP_HORIZON_EXHAUSTED')
    if deadline_missed:
        failures.append('CONTROL_DEADLINE_MISSED')
    if current.get('index') != monitor.last_index:
        failures.append('CURRENT_CHECKPOINT_MISMATCH')
    proof = next((p for p in sealed['proofs'] if p['index'] == current.get('index')), None)
    if proof is None or proof['checkpoint_sha256'] != digest(current):
        failures.append('CURRENT_CHECKPOINT_UNPROVEN')
    subset = False
    if proof is not None and query['volume'] is not None:
        # Exact containment: never expand the scanned region with a tolerance.
        subset = all(a <= b <= c <= d for a, b, c, d in zip(
            query['volume']['lower'], proof['lower'], proof['upper'], query['volume']['upper']))
        if not subset:
            failures.append('REMAINING_VOLUME_OUTSIDE_CHECKED_REGION')
    if query['geometry'] is None:
        failures.append('NO_GEOMETRY_QUERY')
    applicable = not failures and subset
    return dict(receipt_sha256=receipt.sha256, query_sha256=sealed['query']['sha256'],
        query_index=query['checkpoint']['index'], current_index=current.get('index'),
        completed_at_s=now_s, wall_age_s=wall_age_s,
        remaining_subset=subset, diagnostic_current=applicable,
        checked_superset_geometry=None if query['geometry'] is None else query['geometry']['status'],
        remaining_volume=None if proof is None else dict(lower=proof['lower'], upper=proof['upper']),
        reasons=list(dict.fromkeys(query['reasons']+failures)), delivery_failures=failures,
        status='HOLD', flight_authorized=False, selected_for_execution=False,
        free_volume_proven=False, navigation_map_update_allowed=False)
