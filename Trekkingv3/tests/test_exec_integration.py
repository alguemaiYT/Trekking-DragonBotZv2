from collections import deque
import importlib.util
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SEND_CSV_PATH = ROOT_DIR / "tests" / "exec" / "send_csv.py"

spec = importlib.util.spec_from_file_location("send_csv", SEND_CSV_PATH)
send_csv = importlib.util.module_from_spec(spec)
sys.modules["send_csv"] = send_csv
spec.loader.exec_module(send_csv)


class FakeSerial:
    def __init__(self, device, baudrate, timeout, write_timeout, responses):
        self.device = device
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.responses = deque(responses)
        self.writes = []

    def write(self, data):
        self.writes.append(data)

    def readline(self):
        if self.responses:
            return self.responses.popleft()
        return b""

    def flush(self):
        return None

    def close(self):
        return None


class FakeSerialModule:
    def __init__(self, responses):
        self.responses = responses
        self.instance = None

    def Serial(self, device, baudrate=115200, timeout=1.0, write_timeout=1.0):
        self.instance = FakeSerial(device, baudrate, timeout, write_timeout, self.responses)
        return self.instance


def test_send_csv_sends_lines_and_counts_acks(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "frame_idx,ts_wallclock_ms,ts_source_ms,source,detected,target_id,cx,cy,err_px,err_norm,err_deg,bbox_h,est_dist_m,avg_score,fps\n"
        "0,1000,900,video,true,1,120,60,+5.0,+0.050,+2.50,80,2.50,0.90,30.0\n"
        "1,1010,910,video,false,,,,,,,,,29.9\n"
    )

    responses = deque([
        b"ACK,OK,1,+0.050,+2.50,1000\n",
        b"NO_ACK\n",
    ])
    fake_serial = FakeSerialModule(responses)

    stats = send_csv.send_csv_file(
        csv_path=csv_path,
        device="/dev/ttyUSB9",
        baud=57600,
        rate_hz=1000.0,
        repeat=False,
        raw=False,
        serial_module=fake_serial,
    )

    assert fake_serial.instance.device == "/dev/ttyUSB9"
    assert fake_serial.instance.baudrate == 57600
    assert fake_serial.instance.writes
    first_line = fake_serial.instance.writes[0].decode("utf-8").strip()
    assert not first_line.startswith("frame_idx")
    assert len(fake_serial.instance.writes) == 2
    assert stats.sent == 2
    assert stats.acks_received == 1
    assert stats.errors == 1
