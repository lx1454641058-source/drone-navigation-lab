"""来源：本项目原创。完整重放两组规划方法及逐例对照条件。"""

import argparse
import json
from pathlib import Path

from .localization_experiment import compare_policies
from .verify_localization import verify


def verify_pair(output: Path):
    before = verify(output/'baseline')
    after = verify(output)
    old = json.loads((output/'baseline/localization_report.json').read_text(encoding='utf-8'))
    new = json.loads((output/'localization_report.json').read_text(encoding='utf-8'))
    if (any(r['policy_mode']!='history' for r in old['results'])
            or any(r['policy_mode']!='reachable' for r in new['results'])):
        raise ValueError('unexpected paired policies')
    expected = compare_policies(new['results'],old['results'])
    if expected!=new['comparisons']:
        raise ValueError('saved comparison differs from replayed results')
    return {'verified':True,'before':before,'after':after,'pairs':len(expected)}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='重放新旧规划实验并验证对照条件')
    parser.add_argument('output',type=Path)
    print(json.dumps(verify_pair(parser.parse_args().output),ensure_ascii=False))
