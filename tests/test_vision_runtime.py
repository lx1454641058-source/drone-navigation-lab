"""来源：本项目原创。新运行器的参数拒绝、进程契约与只读重放边界。"""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from drone_nav.vision_runtime import CPUVisionSession,ROOT,MODEL_SHA
from drone_nav.realvision import sha


class RuntimeTests(unittest.TestCase):
    def config(self,directory,threads=8,input_mode='raw'):
        directory.mkdir()
        (directory/'configuration.json').write_text(json.dumps(dict(threads=threads,input_mode=input_mode,
            worker_sha256=sha(ROOT/'tools/vision_runtime_worker.cjs'),model_sha256=MODEL_SHA)),encoding='utf-8')

    def test_invalid_options_do_not_create_output_or_start_process(self):
        with tempfile.TemporaryDirectory() as d, patch('drone_nav.vision_runtime.subprocess.Popen') as popen:
            output=Path(d)/'sensor'
            for options in (dict(threads=True),dict(threads=3),dict(threads=0),dict(threads=32),
                            dict(input_mode='binary'),dict(timeout_s=0),dict(timeout_s=float('nan'))):
                with self.assertRaises(ValueError): CPUVisionSession(output,**options)
                self.assertFalse(output.exists())
            popen.assert_not_called()

    def test_replay_refuses_configuration_mismatch_before_process(self):
        with tempfile.TemporaryDirectory() as d, patch('drone_nav.vision_runtime.subprocess.Popen') as popen:
            output=Path(d)/'sensor'; self.config(output,4,'gzip')
            with self.assertRaises(ValueError): CPUVisionSession(output,replay=True,threads=8,input_mode='gzip')
            popen.assert_not_called()

    def test_real_worker_handshake_and_close_leave_replay_directory_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'sensor'; self.config(output)
            before={p.name:p.read_bytes() for p in output.iterdir()}
            with CPUVisionSession(output,replay=True) as session:
                self.assertIsNone(session.process.poll())
                self.assertEqual(session.threads,8)
                with self.assertRaises(ValueError): session.detect_packed(None)
            self.assertEqual(session.process.poll(),0)
            self.assertEqual(before,{p.name:p.read_bytes() for p in output.iterdir()})

    def test_worker_itself_rejects_unsupported_options(self):
        result=subprocess.run(['node',str(ROOT/'tools/vision_runtime_worker.cjs')],
            input=json.dumps(dict(threads=3,input_mode='raw',replay=True))+'\n',
            text=True,capture_output=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('unsupported runtime options',json.loads(result.stdout)['error'])


if __name__=='__main__': unittest.main()
