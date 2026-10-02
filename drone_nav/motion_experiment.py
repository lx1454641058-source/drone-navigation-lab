"""来源：本项目原创。运动学实验和同条件探索策略对照；不新增第三方资源。"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .exploration_experiment import SimulatedDepthCamera, audit_motion, exploration_cases
from .motion import MotionConfig
from .timed_navigation import navigate


def motion_cases():
    original = exploration_cases()
    cases = [dict(c, config=MotionConfig(), failure_edge=None, policy_mode='history') for c in original]
    cases[2]['title'] = '03 · 停驻扫描时深度失效'
    for key, title, config, fault in (
        ('moving_failure', '04 · 移动途中故障与制动', MotionConfig(), 0),
        ('long_reaction', '05 · 延迟过大，停止空间不足', MotionConfig(reaction_s=3), None),
        ('short_lifetime', '06 · 观测即将过期，拒绝移动', MotionConfig(free_ttl_s=1), None),
    ):
        cases.append(dict(original[0], key=key, title=title, config=config,
                          failure_edge=fault, policy_mode='history'))
    for original_case in original[:2]:
        cases.append(dict(original_case, key='legacy_'+original_case['key'],
                          title='对照 · '+original_case['title'][5:], config=MotionConfig(),
                          failure_edge=None, policy_mode='legacy'))
    return cases


def run_motion_suite(output: Path):
    results, manifest = [], []
    for case in motion_cases():
        camera = SimulatedDepthCamera(case['world'], case['dropout_tick'], record=True)
        result = navigate(camera, 20, 16, case['start'], case['goal'], config=case['config'],
                          failure_edge=case['failure_edge'], policy_mode=case['policy_mode'])
        result.update(key=case['key'], title=case['title'], world=asdict(case['world']),
                      dropout_tick=case['dropout_tick'], seed=1201,
                      audit=audit_motion(case['world'], result))
        raw_path = case['key']+'-observations.json'
        blob = json.dumps(camera.frames, allow_nan=False, separators=(',', ':')).encode('utf-8')
        (output/raw_path).write_bytes(blob)
        manifest.append({'path':raw_path, 'frames':len(camera.frames),
                         'sha256':hashlib.sha256(blob).hexdigest()})
        results.append(result)
    comparisons = []
    for key in ('around_building', 'blocked_world'):
        new = next(r for r in results if r['key'] == key)
        old = next(r for r in results if r['key'] == 'legacy_'+key)
        comparisons.append({'scenario':key, 'history_terminal':new['terminal_state'],
                            'legacy_terminal':old['terminal_state'], 'history_m':new['distance_m'],
                            'legacy_m':old['distance_m'], 'distance_delta_m':new['distance_m']-old['distance_m'],
                            'note':'相同场景、相机、种子、运动和时效配置；仅探索选点策略不同。两个开发场景不能代表泛化性能。'})
    report = {'version':'0.5.0', 'mode':'motion', 'results':results, 'comparisons':comparisons}
    (output/'motion_report.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (output/'observations_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    template = (Path(__file__).parent/'motion_lab.html').read_text(encoding='utf-8')
    (output/'demo.html').write_text(template.replace('__MOTION_DATA__', json.dumps(
        report, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')), encoding='utf-8')
    return report
