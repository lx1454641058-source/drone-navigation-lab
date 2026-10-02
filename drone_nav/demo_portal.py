"""来源：本项目原创。仅在本机展示已保存的毕设实验，不从网页执行任务。"""
import argparse
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
import webbrowser

PROJECT = Path(__file__).resolve().parents[1]
CATALOG = (
    ('mission', '统一任务流程', 'supervised-mission-01', '巡航、下降、中止、恢复与停桨联锁。先查看正常接地，再对照水面和探针故障。'),
    ('navigation', '障碍观测与巡航', 'run-v10-coupled', '查看观测地图、绕行路线、实际运动，以及未确认到达的结果。'),
    ('descent', '末端下降与接地', 'run-v11-descent-validated', '查看下降过程、水面拒降、证据过期和停桨后接地未确认。'),
    ('vision', '真实航拍识别', 'tinyformer-probe-01', '对照原图、标注、检测框、漏检与误检。真实识别仍是离线实验。'),
    ('rescan', '悬停与相机复查', 'physical-rescan-02', '转动相机补充观察；比较发现细障碍和仍然漏采的案例。'),
    ('recovery', '异常后的连续恢复', 'hover-recovery-01', '查看连续稳定条件、超时、反馈错误和恢复失败。'),
    ('routes', '连续路线与转弯', 'route-continuation-01', '查看四个方向及两条转弯路线，比较缓存超限与改进后的实际轨迹。'),
    ('building', '建筑绕行与途中故障', 'building-route-01', '六组开发对照：建筑遮挡、补充观察，以及途中测距失效和航点未执行。验证状态见当前成果清单。'),
)
DOCUMENTS = ('RESULTS_CATALOG.md', 'RESULTS_BRIEF.md', 'RESULTS_OVERVIEW.md', 'DEMO_GUIDE.md')


def within(path, root):
    """Resolve first so that symlinks and Windows path forms cannot escape."""
    resolved = path.resolve()
    return resolved if resolved.is_relative_to(root.resolve()) else None


def catalog(project=PROJECT, mission_run=None):
    entries = []
    for key, title, folder, description in CATALOG:
        directory = project / 'work' / folder
        if key == 'mission' and mission_run is not None:
            directory = Path(mission_run).resolve()
            if within(directory, project / 'work') is None:
                raise ValueError('任务归档必须位于本项目 work 目录内')
            report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            if report.get('kind') != 'supervised-mission':
                raise ValueError('指定目录不是统一任务实验')
        directory = within(directory, project / 'work')
        available = directory is not None and (directory / 'demo.html').is_file()
        entries.append(dict(key=key, title=title, description=description,
                            directory=directory, available=available))
    return entries


def page(entries):
    cards = []
    for i, item in enumerate(entries, 1):
        action = (f'<a href="/runs/{item["key"]}/demo.html">打开演示 <span aria-hidden="true">↗</span></a>'
                  if item['available'] else '<span class="missing">归档缺失，请按运行说明生成或恢复</span>')
        folder = item['directory'].name if item['directory'] else '目录不可用'
        cards.append(f'<article><div class="number">0{i}</div><h2>{escape(item["title"])}</h2>'
                     f'<p>{escape(item["description"])}</p>{action}<small>{escape(folder)}</small></article>')
    template = (PROJECT / 'drone_nav/project_portal.html').read_text(encoding='utf-8')
    return template.replace('__CARDS__', ''.join(cards)).encode('utf-8')


class PortalHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, project=PROJECT, entries=None, **kwargs):
        self.project = project.resolve()
        self.entries = entries if entries is not None else catalog(project)
        super().__init__(*args, directory=str(project), **kwargs)

    def mapped_file(self):
        # Decode once, then reject ambiguous separators and drive syntax on Windows.
        path = unquote(urlsplit(self.path).path)
        if '\\' in path or ':' in path or '\0' in path:
            return None
        pieces = path.strip('/').split('/')
        if any(p in ('..', '.') for p in pieces):
            return None
        if len(pieces) == 2 and pieces[0] == 'docs' and pieces[1] in DOCUMENTS:
            return within(self.project / 'docs' / pieces[1], self.project / 'docs')
        if len(pieces) >= 3 and pieces[0] == 'runs':
            item = next((x for x in self.entries if x['key'] == pieces[1]), None)
            if item and item['available']:
                return within(item['directory'].joinpath(*pieces[2:]), item['directory'])
        return None

    def send_head(self):
        if urlsplit(self.path).path == '/':
            from io import BytesIO
            data = page(self.entries)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            return BytesIO(data)
        file = self.mapped_file()
        if file is None or not file.is_file():
            self.send_error(404, 'Saved artifact not found')
            return None
        try:
            stream = file.open('rb')
        except OSError:
            self.send_error(404, 'Saved artifact not readable')
            return None
        self.send_response(200)
        content_type = self.guess_type(str(file))
        if file.suffix == '.md':
            content_type = 'text/plain'
        if content_type.startswith('text/'):
            content_type += '; charset=utf-8'
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(file.stat().st_size))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        return stream


def make_server(project=PROJECT, port=8798, mission_run=None):
    entries = catalog(project, mission_run)
    return ThreadingHTTPServer(('127.0.0.1', port), partial(
        PortalHandler, project=project, entries=entries))


def main():
    parser = argparse.ArgumentParser(description='无人机毕设本地演示入口')
    parser.add_argument('--port', type=int, default=8798)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--mission-run', type=Path)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('端口必须为 0 到 65535；0 表示自动选择')
    try:
        server = make_server(port=args.port, mission_run=args.mission_run)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'无法启动演示：{exc}\n端口占用可用 --port 0；不会停止已有服务。\n')
    with server:
        url = f'http://127.0.0.1:{server.server_port}/'
        print(f'演示首页：{url}\n仅查看已保存结果；按 Ctrl+C 停止服务。', flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
