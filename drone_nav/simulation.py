"""来源：本项目原创。复用已验证的连续观测车辆，统一配置、运行、重放和查看。"""
import argparse
from datetime import datetime
from functools import partial
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
import webbrowser

from tools import building_route_experiment as engine

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'work/building-route-01'
OWN = ('drone_nav/simulation.py', 'tests/test_simulation.py',
       'docs/SIMULATION_PROTOCOL.md', 'simulate.ps1')
PRESETS = {
    'open': ('开放场地连续路线', [7, 8]),
    'building': ('建筑绕行', [9, 8]),
    'sensor-fault': ('途中测距失效', [9, 8]),
    'control-fault': ('第二个航点未执行', [9, 8]),
}


def configuration(scenario, goal=None, max_ticks=24):
    if scenario not in PRESETS:
        raise ValueError('未知场景')
    if type(max_ticks) is not int or not 1 <= max_ticks <= 64:
        raise ValueError('导航步数上限必须为 1 到 64 的整数')
    goal = list(PRESETS[scenario][1]) if goal is None else goal
    if (not isinstance(goal, (list, tuple)) or len(goal) != 2
            or any(type(x) is not int for x in goal)
            or not 1 <= goal[0] <= 18 or not 1 <= goal[1] <= 14):
        raise ValueError('目标格必须为两个整数：x=1..18，y=1..14')
    return dict(scenario=scenario, goal=list(goal), max_ticks=max_ticks)


def make_spec(config):
    expected = configuration(**config)
    if config != expected:
        raise ValueError('配置必须使用规范化结构')
    scenario = config['scenario']
    spec = engine.base_spec()
    spec.update(key=scenario, title=PRESETS[scenario][0], goal=config['goal'][:],
                max_ticks=config['max_ticks'], wide_scan=True, level_scan=True,
                mission_goal_probe=True)
    if scenario == 'open':
        spec['buildings'] = []
    elif scenario == 'sensor-fault':
        spec['sensor_after_moves'] = 1
    elif scenario == 'control-fault':
        spec['ignore_move'] = 1
    return spec


def dependencies():
    """Refuse changed experiment dependencies instead of silently changing a baseline."""
    engine.check_manifest(PARENT)
    parent = json.loads((PARENT / 'report.json').read_text(encoding='utf-8'))
    sources = {name: engine.sha(ROOT / name) for name in sorted(set(parent['sources']) | set(OWN))}
    if any(sources[name] != digest for name, digest in parent['sources'].items()):
        raise ValueError('已验证的算法来源发生变化；须先单独验证新基线')
    engine.check_manifest(engine.MODEL.parent)
    model = engine.ColorModel.from_dict(json.loads(engine.MODEL.read_text(encoding='utf-8')))
    return sources, engine.sha(PARENT / 'report.json'), model


def render_page(report):
    # The original building page crops to its fixed route. Restore the full map
    # so custom targets anywhere in the allowed grid remain visible.
    html = engine.page(report).decode('utf-8')
    html = html.replace('viewBox="56 112 280 252"', 'viewBox="0 0 560 448"')
    html = html.replace('建筑遮挡与途中异常', '无人机仿真运行结果')
    html = html.replace('建筑挡住直达方向后，如何继续？', '无人机仿真运行结果')
    config = report['configuration']
    description = (f'<p>场景：{escape(PRESETS[config["scenario"]][0])}；'
                   f'起点：(3,8)；目标：{escape(str(config["goal"]))}；'
                   f'导航步数上限：{config["max_ticks"]}。'
                   '这是一次新计算的保存结果，成功与故障终态均原样保留。'
                   '导航步数是扫描/规划循环次数，不是物理计算步数。</p>')
    start = html.index('<p>相同建筑场景')
    end = html.index('</p>', start) + 4
    html = html[:start] + description + ('<p>理想传感器、精确位姿、静态场地和明确起始声明；'
        '真实视觉尚未接控制，未实现完整配送。不保证任意目标可达。</p>') + html[end:]
    return html.encode('utf-8')


def read_archive(folder):
    """Check required files as well as digests; an empty manifest is not verification."""
    folder = Path(folder).resolve()
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    if report.get('kind') != 'unified-simulation' or report.get('schema_version') != 1:
        raise ValueError('不是统一仿真运行归档')
    spec = make_spec(report['configuration'])
    cases = report['cases']
    if len(cases) != 1:
        raise ValueError('每份归档必须恰有一个场景')
    case = cases[0]
    if (case['key'] != spec['key'] or case['title'] != spec['title']
            or case['input'] != spec['key'] + '/raw.json.gz'
            or case['summary']['spec'] != spec):
        raise ValueError('配置、场景和输入路径不一致')
    required = {'report.json', 'demo.html', 'protocol.md', case['input']}
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest) != required:
        raise ValueError('归档文件清单不完整或含非预期文件')
    engine.check_manifest(folder)
    return folder, report, manifest


def verify(folder):
    folder, report, manifest = read_archive(folder)
    sources, parent_hash, model = dependencies()
    if report['sources'] != sources or report['parent_sha256'] != parent_hash:
        raise ValueError('运行源码或父实验不一致')
    engine.verify_case(report['cases'][0], model, folder)
    if render_page(report) != (folder / 'demo.html').read_bytes():
        raise ValueError('页面与报告不一致')
    result = dict(verified=True, files=len(manifest), sources=len(sources),
                  frames=report['cases'][0]['summary']['frames'],
                  steps=report['cases'][0]['summary']['steps'])
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run(config, output):
    spec = make_spec(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('拒绝覆盖已有目录：' + str(output))
    sources, parent_hash, model = dependencies()
    output.mkdir(parents=True, exist_ok=False)
    print('开始仿真：' + str(output), flush=True)
    case = engine.archive_case(spec, model, output)
    if sources != {name: engine.sha(ROOT / name) for name in sources}:
        raise ValueError('运行期间源码发生变化，保留已有输出但不标记完成')
    report = dict(kind='unified-simulation', schema_version=1,
                  configuration=config, sources=sources, parent_sha256=parent_hash, cases=[case])
    engine.dump(output / 'report.json', report)
    engine.put(output / 'demo.html', render_page(report))
    engine.put(output / 'protocol.md', (ROOT / 'docs/SIMULATION_PROTOCOL.md').read_bytes())
    engine.dump(output / 'manifest.json', {
        p.relative_to(output).as_posix(): engine.sha(p)
        for p in output.rglob('*') if p.is_file()})
    print('仿真终态：' + case['summary']['result']['state'] + '；开始完整重放核验。', flush=True)
    verify(output)
    print('核验完成，结果：' + str(output / 'demo.html'), flush=True)
    return output


class ResultHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, folder, allowed, **kwargs):
        self.folder, self.allowed = folder, allowed
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path).lstrip('/') or 'demo.html'
        file = (self.folder / path).resolve()
        if path not in self.allowed or not file.is_relative_to(self.folder) or not file.is_file():
            self.send_error(404)
            return
        types = {'.html': 'text/html; charset=utf-8', '.json': 'application/json',
                 '.md': 'text/plain; charset=utf-8', '.gz': 'application/gzip'}
        self.send_response(200)
        self.send_header('Content-Type', types.get(file.suffix, 'application/octet-stream'))
        self.send_header('Content-Length', str(file.stat().st_size))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if self.command != 'HEAD':
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    self.wfile.write(block)

    do_HEAD = do_GET


def view(folder, port=0, open_browser=False):
    folder, _, manifest = read_archive(folder)
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('端口须为 0..65535')
    with ThreadingHTTPServer(('127.0.0.1', port), partial(
            ResultHandler, folder=folder, allowed=set(manifest) | {'manifest.json'})) as server:
        url = f'http://127.0.0.1:{server.server_port}/'
        print('保存结果：' + url + '\n仅检查归档摘要；完整重放请使用 verify。Ctrl+C 停止服务。', flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description='无人机仿真：选场景、运行并核验、查看保存结果')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='列出预设，不运行仿真')
    p = commands.add_parser('run', help='新建仿真，并自动完成重放核验')
    p.add_argument('--scenario', choices=PRESETS, default='building')
    p.add_argument('--goal', type=int, nargs=2, metavar=('X', 'Y'))
    p.add_argument('--max-ticks', type=int, default=24)
    p.add_argument('--output', type=Path)
    p.add_argument('--open', action='store_true', help='核验后启动结果服务并打开浏览器')
    p = commands.add_parser('verify', help='完整重放已有统一运行归档')
    p.add_argument('--input', type=Path, required=True)
    p = commands.add_parser('view', help='检查摘要并查看保存结果，不重新计算')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--port', type=int, default=0)
    p.add_argument('--open', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'list':
            for key, (title, goal) in PRESETS.items():
                print(f'{key}: {title}；默认目标 {goal}')
        elif args.command == 'verify':
            verify(args.input)
        elif args.command == 'view':
            view(args.input, args.port, args.open)
        else:
            config = configuration(args.scenario, args.goal, args.max_ticks)
            output = args.output or ROOT / 'work' / datetime.now().strftime('simulation-%Y%m%d-%H%M%S-%f')
            output = run(config, output)
            if args.open:
                view(output, open_browser=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'未完成：{exc}\n已有输出保留；不会自动重试或覆盖。\n')


if __name__ == '__main__':
    main()
