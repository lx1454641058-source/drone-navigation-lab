"""来源：本项目原创。独立构造 ROS 消息检查原始字节读取。"""
import bz2
import struct
import tempfile
import unittest
from pathlib import Path
from tools.tum_bag_subset import fields, images, records, stamp


def sized(data): return struct.pack('<I',len(data))+data
def header(**values): return b''.join(sized(k.encode()+b'='+v) for k,v in values.items())
def record(h,d): return sized(h)+sized(d)


class BagInputTests(unittest.TestCase):
    def test_binary_fields_and_truncation(self):
        self.assertEqual(fields(header(binary=b'=\x00')), {b'binary':b'=\x00'})
        with self.assertRaises(ValueError): fields(sized(b'x=1')+sized(b'x=2'))
        with self.assertRaises(ValueError): list(records(record(header(op=b'\x02'),b'abc')[:-1]))
        self.assertAlmostEqual(stamp(struct.pack('<II',3,200000000)),3.2)
        with self.assertRaises(ValueError): stamp(struct.pack('<II',3,1000000000))

    def test_lossless_float_depth_and_header_capture_time(self):
        body = (struct.pack('<III',7,10,250000000)+sized(b'camera')+
            struct.pack('<II',480,640)+sized(b'32FC1')+b'\0'+struct.pack('<I',2560)+
            sized(struct.pack('<f',1.25)*(640*480)))
        message=record(header(op=b'\x02',conn=struct.pack('<I',5),time=struct.pack('<II',10,260000000)),body)
        chunk=record(header(op=b'\x05',compression=b'bz2',size=struct.pack('<I',len(message))),bz2.compress(message))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'chunk.bin'; p.write_bytes(chunk)
            result=images(p,{5:dict(topic='/camera/depth/image',type='sensor_msgs/Image')})[0]
            self.assertEqual(result['captured_at_s'],10.25)
            self.assertEqual(result['bag_at_s'],10.26)
            self.assertEqual(result['raw_message'],body)
            self.assertEqual(result['pixels'][:4],struct.pack('<f',1.25))
            self.assertEqual(result['encoding'],'32FC1')


if __name__=='__main__': unittest.main()
