"""来源：本项目原创。四种布局与四种定位条件的开发矩阵，全部结果原样保留。"""

from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

from .localization import LocalizationBudget, MislocalizedCamera, PositionError, audit_localization
from .raycast import Box, Surface, World
from .timed_navigation import navigate


def localization_cases():
    ground = Surface('ground',0,(-40,60,-40,56))
    layouts = [
        ('open','开阔场地',()),
        ('building','中央建筑',(Box('building',3,(8,6,0),(10,10,6)),)),
        ('staggered','错开建筑',(Box('west',3,(7,3,0),(9,9,6)),Box('east',3,(12,9,0),(14,13,6)))),
        ('corridor','建筑间通道',(Box('south',3,(7,0,0),(13,5,6)),Box('north',3,(7,11,0),(13,16,6)))),
    ]
    conditions = [
        ('ideal','无误差',PositionError(),LocalizationBudget()),
        ('unaware','误差未计入',PositionError(.45),LocalizationBudget()),
        ('bounded','预留误差空间',PositionError(.45),LocalizationBudget(.45)),
        ('drift','缓慢变化误差',PositionError(.25,'drift'),LocalizationBudget(.25)),
    ]
    return [dict(key=layout+'_'+condition,title=title+' / '+label,layout=layout,condition=condition,
                 world=World((ground,*boxes)),error=error,budget=budget,start=(3,8),goal=(16,8))
            for layout,title,boxes in layouts for condition,label,error,budget in conditions]


def summarize(results):
    summary = []
    for condition in ('ideal','unaware','bounded','drift'):
        runs = [r for r in results if r['condition']==condition]
        summary.append({'condition':condition,'runs':len(runs),
                        'reported_arrivals':sum(r['terminal_state']=='ARRIVED_WAYPOINT' for r in runs),
                        'actual_arrivals':sum(r['audit']['inside_goal_tolerance'] for r in runs),
                        'false_arrivals':sum(r['audit']['false_arrival'] for r in runs),
                        'collisions':sum(r['audit']['collision_segments']+r['audit']['endpoint_collisions'] for r in runs),
                        'boundary_violations':sum(r['audit']['boundary_violations'] for r in runs)})
    return summary


def compare_policies(results, baseline):
    """逐例核对控制条件，防止把不同布局或误差强度混为策略提升。"""
    old = {r['key']:r for r in baseline}
    if len(old)!=len(baseline) or set(old)!={r['key'] for r in results} or len(results)!=len(old):
        raise ValueError('paired experiment keys do not match')
    comparisons = []
    for result in results:
        previous = old[result['key']]
        for field in ('world','error_model','localization_budget','config','intrinsics','start','goal','max_ticks','seed'):
            if result[field]!=previous[field]:
                raise ValueError('paired conditions differ: '+field)
        comparisons.append({'key':result['key'],'title':result['title'],
                            'before_terminal':previous['terminal_state'],'after_terminal':result['terminal_state'],
                            'before_distance_m':previous['distance_m'],'after_distance_m':result['distance_m'],
                            'before_actual_arrival':previous['audit']['inside_goal_tolerance'],
                            'after_actual_arrival':result['audit']['inside_goal_tolerance'],
                            'before_false_arrival':previous['audit']['false_arrival'],
                            'after_false_arrival':result['audit']['false_arrival']})
    return comparisons


def run_localization_suite(output: Path, *, policy_mode='history', baseline=None):
    results, manifest = [], []
    for case in localization_cases():
        camera = MislocalizedCamera(case['world'],case['error'],record=True)
        result = navigate(camera,20,16,case['start'],case['goal'],localization=case['budget'],policy_mode=policy_mode)
        result.update(key=case['key'],title=case['title'],layout=case['layout'],condition=case['condition'],
                      world=asdict(case['world']),error_model=asdict(case['error']),seed=1201,
                      audit=audit_localization(case['world'],result,case['error'],case['budget']))
        name = case['key']+'-observations.json.gz'
        raw = json.dumps(camera.frames,allow_nan=False,separators=(',',':')).encode('utf-8')
        blob = gzip.compress(raw,compresslevel=6,mtime=0)
        (output/name).write_bytes(blob)
        manifest.append({'path':name,'frames':len(camera.frames),'sha256':hashlib.sha256(blob).hexdigest()})
        results.append(result)
        print(f"{case['key']}: {result['terminal_state']} | reported={result['distance_m']:.2f} m | goal error={result['audit']['final_goal_error_m']:.3f} m",flush=True)
    report = {'version':'0.7.0' if policy_mode=='reachable' else '0.6.0',
              'mode':'planning' if policy_mode=='reachable' else 'localization',
              'results':results,'summary':summarize(results),
              'scope':'四个原创开发布局，不是独立冻结测试集；全部 16 次运行保留。'}
    if baseline is not None:
        report['comparisons'] = compare_policies(results,baseline)
    serialized = json.dumps(report,ensure_ascii=False,allow_nan=False)
    (output/'localization_report.json').write_text(serialized,encoding='utf-8')
    (output/'observations_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    template = (Path(__file__).parent/'localization_lab.html').read_text(encoding='utf-8')
    (output/'demo.html').write_text(template.replace('__LOCALIZATION_DATA__',serialized.replace('<','\\u003c')),encoding='utf-8')
    return report


def run_planning_suite(output: Path):
    baseline_path = output/'baseline'
    baseline_path.mkdir(exist_ok=False)
    print('Running original history policy...',flush=True)
    baseline = run_localization_suite(baseline_path)
    print('Running stopping-feasible edge policy...',flush=True)
    return run_localization_suite(output,policy_mode='reachable',baseline=baseline['results'])
