"""来源：本项目原创。演示资源可达、缺失状态和路径隔离的 HTTP 检查。"""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from drone_nav.demo_portal import CATALOG, catalog, make_server


class PortalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'work' / CATALOG[0][2]
        self.folder.mkdir(parents=True)
        (self.folder / 'demo.html').write_text('<h1>统一任务</h1>', encoding='utf-8')
        (self.folder / 'report.json').write_text(json.dumps({'kind': 'supervised-mission'}))
        (self.folder / 'frames').mkdir()
        (self.folder / 'frames/a.png').write_bytes(b'example-image')
        (self.root / 'private.txt').write_text('not an artifact')
        (self.root / 'docs').mkdir()
        (self.root / 'docs/DEMO_GUIDE.md').write_text('中文说明', encoding='utf-8')
        for key in ('routes', 'building'):
            item = next(item for item in CATALOG if item[0] == key)
            folder = self.root / 'work' / item[2]
            folder.mkdir(parents=True)
            (folder / 'demo.html').write_text(item[1], encoding='utf-8')
        (self.root / 'docs/RESULTS_CATALOG.md').write_text('当前成果', encoding='utf-8')
        self.server = make_server(self.root, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, method='GET'):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            conn.request(method, path)
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_home_links_only_available_archives(self):
        code, _, data = self.request('/')
        self.assertEqual(code, 200)
        text = data.decode('utf-8')
        self.assertIn('href="/runs/mission/demo.html"', text)
        self.assertNotIn('href="/runs/vision/demo.html"', text)
        self.assertIn('归档缺失', text)

    def test_saved_pages_relative_images_and_unicode_docs(self):
        self.assertIn('统一任务', self.request('/runs/mission/demo.html')[2].decode('utf-8'))
        self.assertEqual(self.request('/runs/mission/frames/a.png')[2], b'example-image')
        code, headers, data = self.request('/docs/DEMO_GUIDE.md')
        self.assertEqual(code, 200)
        self.assertIn('charset=utf-8', headers['Content-Type'])
        self.assertEqual(data.decode('utf-8'), '中文说明')

    def test_research_archives_and_current_catalog_are_reachable(self):
        # New entries use the same isolated HTTP mapping as the original demos.
        code, _, data = self.request('/')
        self.assertEqual(code, 200)
        for key in ('routes', 'building'):
            self.assertIn(f'href="/runs/{key}/demo.html"', data.decode('utf-8'))
            self.assertEqual(self.request(f'/runs/{key}/demo.html')[0], 200)
        self.assertIn('当前成果', self.request('/docs/RESULTS_CATALOG.md')[2].decode('utf-8'))

    def test_unlisted_paths_traversal_and_directories_not_served(self):
        for path in ('/private.txt', '/docs/other.md', '/runs/mission/',
                     '/runs/mission/../../private.txt', '/runs/mission/%2e%2e/%2e%2e/private.txt',
                     '/runs/mission/..%5c..%5cprivate.txt', '/runs/mission/C:/private.txt',
                     '/runs/mission/%00.txt', '/runs/unknown/demo.html'):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)

    def test_no_mutation_and_head_has_no_body(self):
        self.assertEqual(self.request('/', 'POST')[0], 501)
        code, headers, data = self.request('/runs/mission/demo.html', 'HEAD')
        self.assertEqual(code, 200)
        self.assertGreater(int(headers['Content-Length']), 0)
        self.assertEqual(data, b'')

    def test_mission_override_must_be_local_mission_archive(self):
        entries = catalog(self.root, self.folder)
        self.assertTrue(entries[0]['available'])
        with self.assertRaises(ValueError):
            catalog(self.root, self.root)
        (self.folder / 'report.json').write_text('{"kind":"other"}')
        with self.assertRaises(ValueError):
            catalog(self.root, self.folder)

    def test_symlink_cannot_escape_archive(self):
        link = self.folder / 'outside.txt'
        try:
            link.symlink_to(self.root / 'private.txt')
        except OSError as exc:
            self.skipTest(f'Symlink unavailable: {exc}')
        self.assertEqual(self.request('/runs/mission/outside.txt')[0], 404)


if __name__ == '__main__':
    unittest.main()
