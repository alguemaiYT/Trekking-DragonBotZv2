import importlib.util
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT_DIR = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT_DIR / "versions" / "v4" / "cone_detector_v4.py"


def load_v4_module():
    spec = importlib.util.spec_from_file_location("cone_detector_v4_api_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_test_jpeg_bytes() -> bytes:
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    frame[:] = (0, 140, 255)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def http_request(method: str, url: str, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return resp.status, dict(resp.headers), resp.read()


class FakeApiService:
    def __init__(self):
        self.image_calls = []
        self.video_calls = []
        self.stream_calls = []
        self.stop_calls = []
        self.stream_status_calls = []
        self.frame_calls = []

    def detect_image(self, *, image_bytes=None, source=None, body=None, query=None, headers=None):
        self.image_calls.append(
            {
                "image_bytes": image_bytes,
                "source": source,
                "body": body,
                "query": query or {},
                "headers": headers or {},
            }
        )
        return {
            "ok": True,
            "kind": "image",
            "count": 1,
            "detections": [{"bbox": [3, 4, 40, 50], "conf": 0.91}],
            "status": "inferred",
        }

    def detect_video(self, *, video_bytes=None, source=None, body=None, query=None, headers=None):
        self.video_calls.append(
            {
                "video_bytes": video_bytes,
                "source": source,
                "body": body,
                "query": query or {},
                "headers": headers or {},
            }
        )
        return {
            "ok": True,
            "kind": "video",
            "frames": 12,
            "output_video": "/tmp/fake_api_output.mp4",
            "logs": ["frames=12", "avg_infer_ms=4.20"],
        }

    def start_stream(self, *, source, body=None, query=None, headers=None):
        self.stream_calls.append(
            {
                "source": source,
                "body": body,
                "query": query or {},
                "headers": headers or {},
            }
        )
        return {
            "ok": True,
            "stream_id": "stream-123",
            "running": True,
            "source": source,
        }

    def get_stream_status(self, stream_id: str):
        self.stream_status_calls.append(stream_id)
        if stream_id != "stream-123":
            raise KeyError(stream_id)
        return {
            "ok": True,
            "stream_id": stream_id,
            "running": True,
            "frames_processed": 8,
            "last_status": "inferred",
            "last_detections": [{"bbox": [5, 6, 20, 30], "conf": 0.88}],
        }

    def get_stream_frame(self, stream_id: str) -> bytes:
        self.frame_calls.append(stream_id)
        if stream_id != "stream-123":
            raise KeyError(stream_id)
        return make_test_jpeg_bytes()

    def stop_stream(self, stream_id: str):
        self.stop_calls.append(stream_id)
        if stream_id != "stream-123":
            raise KeyError(stream_id)
        return {
            "ok": True,
            "stream_id": stream_id,
            "running": False,
            "frames_processed": 10,
            "stopped": True,
        }


@pytest.fixture
def v4_module():
    return load_v4_module()


@pytest.fixture
def api_server(v4_module):
    service = FakeApiService()
    server = v4_module.create_api_server("127.0.0.1:0", service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"

    for _ in range(50):
        try:
            http_request("GET", f"{base_url}/healthz")
            break
        except Exception:
            time.sleep(0.02)
    else:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        pytest.fail("API server did not become ready in time")

    try:
        yield service, base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_arg_parser_accepts_api_without_source(v4_module):
    parser = v4_module.build_arg_parser()
    args = parser.parse_args(["--api", "127.0.0.1:9820"])

    assert args.api == "127.0.0.1:9820"
    assert args.source in ("", None)


def test_resolve_client_source_ref_uses_client_cwd(v4_module):
    resolved = v4_module.resolve_client_source_ref("0_image1.png", client_cwd="/tmp/dataset")

    assert resolved == str(Path("/tmp/dataset/0_image1.png").resolve())
    assert v4_module.resolve_client_source_ref("0", client_cwd="/tmp/dataset") == "0"
    assert v4_module.resolve_client_source_ref("rtsp://camera.local/live", client_cwd="/tmp/dataset") == "rtsp://camera.local/live"


def test_resolve_requested_image_output_adds_extension_and_uses_client_cwd(v4_module):
    resolved = v4_module.resolve_requested_image_output(
        "renders/result",
        client_cwd="/tmp/dataset",
        output_format="png",
    )

    assert resolved == Path("/tmp/dataset/renders/result.png").resolve()
    assert v4_module.normalize_image_output_format("jpeg") == ".jpg"


def test_health_endpoint_returns_json(api_server):
    _service, base_url = api_server

    status, headers, body = http_request("GET", f"{base_url}/healthz")
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert payload["ok"] is True


def test_detect_image_accepts_raw_bytes(api_server):
    service, base_url = api_server
    image_bytes = make_test_jpeg_bytes()

    status, headers, body = http_request(
        "POST",
        f"{base_url}/detect/image?return_image=1",
        data=image_bytes,
        headers={"Content-Type": "image/jpeg", "X-Filename": "frame.jpg"},
    )
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert payload["kind"] == "image"
    assert payload["count"] == 1
    assert service.image_calls
    assert service.image_calls[0]["image_bytes"] == image_bytes
    assert service.image_calls[0]["query"]["return_image"] == ["1"]


def test_detect_video_accepts_json_source(api_server):
    service, base_url = api_server
    request_body = {"source": "rtsp://camera.local/live", "output": "/tmp/out.mp4", "max_frames": 25}

    status, headers, body = http_request(
        "POST",
        f"{base_url}/detect/video",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    payload = json.loads(body.decode("utf-8"))

    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert payload["kind"] == "video"
    assert payload["frames"] == 12
    assert service.video_calls
    assert service.video_calls[0]["source"] == "rtsp://camera.local/live"
    assert service.video_calls[0]["body"]["max_frames"] == 25


def test_stream_lifecycle_supports_start_status_frame_and_stop(api_server):
    service, base_url = api_server
    start_body = {"source": "0", "max_frames": 60, "output": "/tmp/live.mp4"}

    status, _headers, body = http_request(
        "POST",
        f"{base_url}/streams/start",
        data=json.dumps(start_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = json.loads(body.decode("utf-8"))

    assert status == 200
    assert started["stream_id"] == "stream-123"
    assert service.stream_calls[0]["source"] == "0"

    status, headers, body = http_request("GET", f"{base_url}/streams/stream-123")
    stream_status = json.loads(body.decode("utf-8"))

    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert stream_status["running"] is True
    assert stream_status["frames_processed"] == 8

    status, headers, frame_bytes = http_request("GET", f"{base_url}/streams/stream-123/frame")
    assert status == 200
    assert headers["Content-Type"].startswith("image/jpeg")
    assert frame_bytes[:2] == b"\xff\xd8"
    assert service.frame_calls == ["stream-123"]

    status, headers, body = http_request("POST", f"{base_url}/streams/stream-123/stop")
    stopped = json.loads(body.decode("utf-8"))

    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert stopped["running"] is False
    assert stopped["stopped"] is True
    assert service.stop_calls == ["stream-123"]


def test_unknown_stream_returns_404_json(api_server):
    _service, base_url = api_server

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        http_request("GET", f"{base_url}/streams/missing-stream")

    assert excinfo.value.code == 404
    payload = json.loads(excinfo.value.read().decode("utf-8"))
    assert payload["ok"] is False
    assert "missing-stream" in payload["error"]
