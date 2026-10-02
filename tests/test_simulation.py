"""来源：本项目原创。统一入口的参数、归档、失败传播及只读显示契约。"""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from functools import partial
from http.server import ThreadingHTTPServer

from drone_nav import simulation as sim


class SimulationTests(unittest.TestCase):
    def fixture(self, root):
        config = sim.configuration('open', [18, 14], 1)
        spec = sim.make_spec(config)
        case = dict(key='open', title=spec['title'], input='open/raw.json.gz',
                    summary=dict(spec=spec))
        report = dict(kind='unified-simulation', schema_version=1, configuration=config,
                      sources={}, parent_sha256='test', cases=[case])
        (root / 'open').mkdir()
        (root / 'open/raw.json.gz').write_bytes(b'test fixture only')
        (root / 'report.json').write_text(json.dumps(report), encoding='utf-8')
        (root / 'demo.html').write_text('仿真结果', encoding='utf-8')
        (root / 'protocol.md').write_text('测试协议', encoding='utf-8')
        manifest = {p.relative_to(root).as_posix(): sim.engine.sha(p)
                    for p in root.rglob('*') if p.is_file()}
        (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return report, manifest

    def test_bad_configuration_rejected_before_any_native_work(self):
        for goal in ([True, 8], [0, 8], [19, 8], [3, 15], [3.1, 8], [3], '3,8'):
            with self.subTest(goal=goal), self.assertRaises(ValueError):
                sim.configuration('open', goal)
        for limit in (True, 0, 65, 1.1):
            with self.assertRaises(ValueError):
                sim.configuration('open', max_ticks=limit)
        with self.assertRaises(ValueError):
            sim.configuration('missing')

    def test_existing_fixed_conditions_reused_without_relaxation(self):
        old = {s['key']: s for s in sim.engine.specs()}
        for preset, key in [('building', 'mission-goal'), ('sensor-fault', 'depth-stage'),
                            ('control-fault', 'ignored-stage')]:
            spec = sim.make_spec(sim.configuration(preset))
            original = dict(old[key], key=spec['key'], title=spec['title'])
            self.assertEqual(spec, original)
        changed = sim.make_spec(sim.configuration('building', [18, 14], 1))
        self.assertEqual(changed['goal'], [18, 14])
        self.assertEqual(changed['max_ticks'], 1)
        self.assertEqual(changed['expiry'], 12.)

    def test_existing_directory_and_bad_config_do_not_start_simulation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(sim, 'dependencies') as dep:
            marker = Path(folder) / 'user.txt'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                sim.run(sim.configuration('open'), folder)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')
            with self.assertRaises(ValueError):
                sim.run(dict(scenario='open', goal=[999, 8], max_ticks=24), Path(folder) / 'new')
            dep.assert_not_called()

    def test_missing_manifest_items_and_changed_data_are_not_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, manifest = self.fixture(root)
            sim.read_archive(root)
            (root / 'manifest.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                sim.read_archive(root)
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            (root / 'demo.html').write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                sim.read_archive(root)

    def test_mismatched_config_or_input_cannot_reach_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report, _ = self.fixture(root)
            report['cases'][0]['input'] = '../private/raw.json.gz'
            (root / 'report.json').write_text(json.dumps(report), encoding='utf-8')
            with patch.object(sim.engine, 'verify_case') as replay, self.assertRaises(ValueError):
                sim.verify(root)
            replay.assert_not_called()

    def test_failed_run_does_not_open_a_result(self):
        with patch.object(sim, 'run', side_effect=ValueError('replay failed')), \
                patch.object(sim, 'view') as view:
            with self.assertRaises(SystemExit) as result:
                sim.main(['run', '--scenario', 'open', '--open'])
            self.assertEqual(result.exception.code, 1)
            view.assert_not_called()

    def test_custom_target_is_not_cropped_out_of_page(self):
        config = sim.configuration('open', [18, 14], 1)
        spec = sim.make_spec(config)
        summary = dict(spec=spec, result=dict(navigation=dict(trace=[]), start=[3, 8],
            state='ABORTED_MOTION_STABLE', confirmed_cell=[3, 8], original_reason='BUDGET'),
            cache=dict(peak_frames=0, retired=[]), history=[dict(position=[3.5, 8.5, 3.5])],
            frames=0, steps=0, audit={}, probes=[])
        html = sim.render_page(dict(configuration=config, cases=[dict(title=spec['title'], summary=summary)])).decode('utf-8')
        self.assertIn('viewBox="0 0 560 448"', html)
        self.assertNotIn('viewBox="56 112 280 252"', html)
        self.assertIn('cx="518.0"', html)
        self.assertIn('[18, 14]', html)
        self.assertNotIn('相同建筑场景对照', html)

    def test_view_serves_only_manifest_files_and_head_without_body(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, manifest = self.fixture(root)
            (root / 'private.txt').write_text('private', encoding='utf-8')
            with ThreadingHTTPServer(('127.0.0.1', 0), partial(
                    sim.ResultHandler, folder=root.resolve(), allowed=set(manifest) | {'manifest.json'})) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    for method, path, status in [('GET', '/', 200), ('HEAD', '/', 200),
                            ('GET', '/private.txt', 404), ('GET', '/open/', 404),
                            ('GET', '/%2e%2e/private.txt', 404), ('POST', '/', 501)]:
                        conn = http.client.HTTPConnection('127.0.0.1', server.server_port)
                        try:
                            conn.request(method, path)
                            response = conn.getresponse()
                            self.assertEqual(response.status, status)
                            body = response.read()
                            if method == 'HEAD':
                                self.assertEqual(body, b'')
                            if method == 'GET' and path == '/':
                                self.assertIn('仿真结果', body.decode('utf-8'))
                        finally:
                            conn.close()
                finally:
                    server.shutdown()
                    thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
