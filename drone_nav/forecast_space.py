"""来源：本项目原创。将已匹配停止预测连接到同参考坐标的深度体积检查。"""
from dataclasses import asdict, dataclass
import re

from .async_stop import ForecastMonitor
from .depth_volume import DepthVolumeInspector, QueryVolume
from .detection_bridge import identifier, number
from .stop_forecast import FrozenForecast, digest


@dataclass(frozen=True)
class SimulationReference:
    """Explicit model/coordinate binding for synthetic data; not real camera calibration."""
    world_frame: str
    clock_id: str
    camera_id: str
    model_sha256: str
    reference_fingerprint: str
    kind: str = 'declared_ideal_simulation_only'

    def __post_init__(self):
        if not all(identifier(v) for v in (self.world_frame, self.clock_id, self.camera_id)):
            raise ValueError('explicit simulation identities required')
        if any(type(v) is not str or re.fullmatch('[0-9a-f]{64}',v) is None
               for v in (self.model_sha256, self.reference_fingerprint)):
            raise ValueError('full model and coordinate-reference digests required')
        if self.kind != 'declared_ideal_simulation_only':
            raise ValueError('real navigation calibration is not implemented')


def depth_identity(inspector):
    if type(inspector) is not DepthVolumeInspector:
        raise TypeError('bound depth inspector required')
    return digest(dict(frame_id=inspector.frame_id, camera_id=inspector.camera_id,
        world_frame=inspector.world_frame, clock_id=inspector.clock_id,
        reference_fingerprint=inspector.reference_fingerprint, pose=asdict(inspector.pose),
        intrinsics=asdict(inspector.intrinsics), depths=inspector.depths, semantics=inspector.semantic,
        captured_at_s=inspector.captured_at_s, available_at_s=inspector.available_at_s,
        valid_until_s=inspector.valid_until_s, source=inspector.depth_source))


def query_stop_space(monitor, current, inspector, reference, *, now_s, clock_id, error_bound_m=.02):
    """Read-only slow-path query. Never run this work inside the 20 ms feedback loop."""
    if type(monitor) is not ForecastMonitor or type(reference) is not SimulationReference:
        raise TypeError('validated stop monitor and explicit simulation reference required')
    if type(inspector) is not DepthVolumeInspector:
        raise TypeError('bound depth inspector required')
    if not number(now_s) or now_s < 0 or not number(error_bound_m) or not 0 <= error_bound_m <= 2:
        raise ValueError('finite time and nonnegative bounded relative error required')
    record = dict(schema='stop-space-query-v1', request_sha256=monitor.request.sha256,
        prediction_sha256=None if monitor.packet is None else monitor.packet.sha256,
        checkpoint=current, reference=asdict(reference), checked_at_s=now_s, clock_id=clock_id,
        monitor_reason=monitor.reason, monitor_index=monitor.last_index,
        depth_identity=None, depth_clock_id=inspector.clock_id, depth_valid_until_s=inspector.valid_until_s,
        geometry=None, volume=None, horizon_end_s=None, reasons=[], status='HOLD',
        free_volume_proven=False, selected_for_execution=False, flight_authorized=False,
        navigation_map_update_allowed=False, physical_navigation_calibrated=False)
    reasons = record['reasons']
    binding = monitor.record['binding']
    if monitor.reason != 'MATCHED' or monitor.points is None:
        reasons.append('STOP_PREDICTION_NOT_MATCHED:'+monitor.reason)
    if (clock_id != binding['clock'] or inspector.clock_id != clock_id or reference.clock_id != clock_id):
        reasons.append('CLOCK_BINDING_MISMATCH')
    if not (binding['world'] == inspector.world_frame == reference.world_frame):
        reasons.append('WORLD_BINDING_MISMATCH')
    if reference.model_sha256 != binding['model_sha256']:
        reasons.append('MODEL_BINDING_MISMATCH')
    if (inspector.reference_fingerprint != reference.reference_fingerprint
        or inspector.camera_id != reference.camera_id):
        reasons.append('CAMERA_REFERENCE_MISMATCH')
    if now_s != current.get('time_s') or current.get('index') != monitor.last_index:
        reasons.append('CURRENT_CHECKPOINT_MISMATCH')
    if reasons:
        return FrozenForecast.create(record)
    # Re-read the immutable packet; a mutable cached monitor point is not the source.
    packet = monitor.packet.record()
    index = current['index']//10
    if not 0 <= index < len(packet['checkpoints'])-1:
        reasons.append('STOP_HORIZON_EXHAUSTED')
        return FrozenForecast.create(record)
    point = packet['checkpoints'][index]
    if any(current.get(key) != value for key,value in point.items() if key != 'remaining'):
        reasons.append('CURRENT_CHECKPOINT_MISMATCH')
        return FrozenForecast.create(record)
    if now_s < inspector.available_at_s:
        reasons.append('DEPTH_NOT_AVAILABLE')
        return FrozenForecast.create(record)
    # Do not scan/hash the full depth grid when cheap identity checks already
    # rejected it. Such a result cannot be published as current evidence.
    record['depth_identity'] = depth_identity(inspector)
    remaining = point['remaining']
    # Map and body position errors can be opposite; add 2E without removing the
    # forecast's original .35 m body radius or .02 m research margin.
    volume = QueryVolume('matched-stop-suffix', reference.world_frame,
        tuple(v-2*error_bound_m for v in remaining['lower_m']),
        tuple(v+2*error_bound_m for v in remaining['upper_m']))
    horizon = packet['checkpoints'][-1]['time_s']
    geometry = inspector.inspect(volume, now_s=now_s, clock_id=clock_id)
    record.update(volume=asdict(volume), geometry=geometry, horizon_end_s=horizon,
                  relative_error_padding_m=2*error_bound_m, forecast_body_radius_m=.35,
                  forecast_margin_m=.02, remaining_horizon_s=horizon-now_s,
                  evidence_remaining_s=inspector.valid_until_s-now_s,
                  expires_before_horizon=horizon > inspector.valid_until_s+1e-9)
    reasons.extend(geometry['reasons'])
    if record['expires_before_horizon']:
        reasons.append('OBSERVATION_EXPIRES_BEFORE_FORECAST_END')
    reasons.extend(('STOP_MODEL_UNCERTAINTY_UNVALIDATED', 'CONTINUOUS_STOP_VOLUME_UNPROVEN'))
    return FrozenForecast.create(record)


def finish_stop_space(query, monitor, current, inspector, *, now_s, clock_id):
    """Reject a slow result if the control context or depth expired during its work."""
    if type(query) is not FrozenForecast or type(monitor) is not ForecastMonitor:
        raise TypeError('immutable query and monitor required')
    record = query.record()
    if record.get('schema') != 'stop-space-query-v1' or not number(now_s) or now_s < record['checked_at_s']:
        raise ValueError('invalid query completion time or schema')
    reasons = list(record['reasons'])
    if clock_id != record['clock_id']:
        reasons.append('COMPLETION_CLOCK_MISMATCH')
    prediction_sha = None if monitor.packet is None else monitor.packet.sha256
    if (monitor.reason != record['monitor_reason'] or monitor.request.sha256 != record['request_sha256']
        or prediction_sha != record['prediction_sha256'] or monitor.last_index != record['monitor_index']
        or monitor.last_index != current.get('index') or current != record['checkpoint']):
        reasons.append('STOP_CONTEXT_CHANGED_DURING_QUERY')
    if record['depth_identity'] is not None and depth_identity(inspector) != record['depth_identity']:
        reasons.append('DEPTH_CONTEXT_CHANGED_DURING_QUERY')
    # Values on unrelated clocks have no comparable age or expiry ordering.
    same_clock = clock_id == record['clock_id'] == record['depth_clock_id'] == current.get('clock') == monitor.record['binding']['clock']
    if same_clock and now_s > record['depth_valid_until_s']+1e-9:
        reasons.append('DEPTH_EXPIRED_DURING_QUERY')
    if same_clock and record['horizon_end_s'] is not None and now_s >= record['horizon_end_s']:
        reasons.append('STOP_HORIZON_EXHAUSTED_DURING_QUERY')
    return dict(query=record, completed_at_s=now_s, reasons=list(dict.fromkeys(reasons)),
        result_current=not any(r.endswith('DURING_QUERY') or r == 'COMPLETION_CLOCK_MISMATCH' for r in reasons)
                       and record['geometry'] is not None,
        status='HOLD', flight_authorized=False, selected_for_execution=False, free_volume_proven=False,
        navigation_map_update_allowed=False)
