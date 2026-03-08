#!/usr/bin/env python3
"""
Detector de cones (YOLO ONNX + OpenCV) para a pasta v4.

Foco:
- usar modelos em v4/yolov26n/{aug,noaug}/{normal,simplified}.onnx
- funcionar em imagem, pasta de imagens, vídeo e câmera
- ter parâmetros úteis para hardware fraco
- manter arquitetura simples para evolução futura
"""
import argparse
import base64
import json
import math
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = SCRIPT_DIR / "yolov26n"


def clamp(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: int = 0
    orange_ratio: float = -1.0

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class ModelSpec:
    family: str
    variant: str
    path: Path
    input_h: int
    input_w: int
    output_shape: Tuple[int, ...]

    @property
    def tag(self) -> str:
        return f"{self.family}/{self.variant} ({self.input_w}x{self.input_h})"


@dataclass
class RuntimePreset:
    family: str
    variant: str
    conf: float
    iou: float
    max_det: int
    det_every: int
    prefilter_enabled: bool
    prefilter_min_ratio: float
    min_box_orange_ratio: float
    threads: int


PRESETS: Dict[str, RuntimePreset] = {
    "fast": RuntimePreset(
        family="aug",
        variant="simplified",
        conf=0.35,
        iou=0.45,
        max_det=80,
        det_every=2,
        prefilter_enabled=True,
        prefilter_min_ratio=0.004,
        min_box_orange_ratio=0.03,
        threads=2,
    ),
    "balanced": RuntimePreset(
        family="aug",
        variant="simplified",
        conf=0.30,
        iou=0.45,
        max_det=120,
        det_every=1,
        prefilter_enabled=True,
        prefilter_min_ratio=0.003,
        min_box_orange_ratio=0.025,
        threads=0,
    ),
    "quality": RuntimePreset(
        family="noaug",
        variant="normal",
        conf=0.25,
        iou=0.50,
        max_det=200,
        det_every=1,
        prefilter_enabled=False,
        prefilter_min_ratio=0.0,
        min_box_orange_ratio=0.015,
        threads=0,
    ),
}


@dataclass
class OverlayContext:
    profile: str
    runtime: RuntimePreset
    use_roi: bool
    roi_norm: Tuple[float, float, float, float]


@dataclass(frozen=True)
class AppContext:
    runtime: RuntimePreset
    detector: Any
    model_tag: str
    overlay_ctx: OverlayContext
    show_orange_ratio: bool
    camera_width: int
    camera_height: int
    buffer_size: int
    drop_grabs: int
    follow_cfg: Optional[Any]
    default_output_dir: Path

    def create_pipeline(self) -> "ConePipeline":
        return ConePipeline(
            detector=self.detector,
            orange_filter=OrangeMaskFilter(),
            det_every=self.runtime.det_every,
            prefilter_enabled=self.runtime.prefilter_enabled,
            prefilter_min_ratio=self.runtime.prefilter_min_ratio,
            min_box_orange_ratio=self.runtime.min_box_orange_ratio,
            use_roi=self.overlay_ctx.use_roi,
            roi_norm=self.overlay_ctx.roi_norm,
            hold_last_on_stride=True,
        )

    def create_follow_estimator(self) -> Optional["ConeFollowErrorEstimator"]:
        if self.follow_cfg is None:
            return None
        return ConeFollowErrorEstimator(cfg=self.follow_cfg)


@dataclass
class FrameProcessResult:
    detections: List[Detection]
    rendered: np.ndarray
    follow_state: Optional[Any]
    status: str
    avg_infer_ms: float
    avg_pipeline_ms: float


@dataclass
class VideoRunResult:
    logs: List[str]
    frame_count: int
    elapsed_s: float
    effective_fps: float
    follow_summary: Optional[Dict[str, float]]


@dataclass
class StreamJobState:
    stream_id: str
    source: str
    output_path: Optional[Path]
    max_frames: Optional[int]
    max_seconds: Optional[float]
    started_at: float
    running: bool = True
    frames_processed: int = 0
    last_timestamp_s: float = 0.0
    last_status: str = "init"
    last_detections: List[Dict[str, Any]] = None
    last_follow: Optional[Dict[str, Any]] = None
    latest_frame_jpeg: Optional[bytes] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    ended_at: Optional[float] = None

    def __post_init__(self) -> None:
        if self.last_detections is None:
            self.last_detections = []


class ModelCatalog:
    def __init__(self, root: Path):
        self.root = root
        self._models: Dict[Tuple[str, str], ModelSpec] = {}
        self._discover()

    def _discover(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Diretório de modelos não encontrado: {self.root}")

        for family_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            family = family_dir.name.lower()
            for onnx_path in sorted(family_dir.glob("*.onnx")):
                variant = onnx_path.stem.lower()
                spec = self._inspect_model(family=family, variant=variant, path=onnx_path)
                self._models[(family, variant)] = spec

        if not self._models:
            raise RuntimeError(f"Nenhum modelo ONNX encontrado em {self.root}")

    @staticmethod
    def _inspect_model(family: str, variant: str, path: Path) -> ModelSpec:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_shape = sess.get_inputs()[0].shape
        output_shape = tuple(int(x) if isinstance(x, int) else -1 for x in sess.get_outputs()[0].shape)
        if len(input_shape) != 4:
            raise RuntimeError(f"Shape de entrada inesperado para {path}: {input_shape}")

        h = int(input_shape[2]) if isinstance(input_shape[2], int) else 640
        w = int(input_shape[3]) if isinstance(input_shape[3], int) else 640

        return ModelSpec(
            family=family,
            variant=variant,
            path=path,
            input_h=h,
            input_w=w,
            output_shape=output_shape,
        )

    def list_models(self) -> List[ModelSpec]:
        return list(sorted(self._models.values(), key=lambda s: (s.family, s.variant)))

    def select(self, family: str, variant: str) -> ModelSpec:
        family = family.lower()
        variant = variant.lower()

        available_families = sorted({k[0] for k in self._models})
        if family == "auto":
            family = "aug" if ("aug" in available_families) else available_families[0]

        family_variants = [k[1] for k in self._models if k[0] == family]
        if not family_variants:
            raise RuntimeError(f"Família '{family}' não encontrada. Disponíveis: {available_families}")

        if variant == "auto":
            if "simplified" in family_variants:
                variant = "simplified"
            elif "normal" in family_variants:
                variant = "normal"
            else:
                variant = sorted(family_variants)[0]

        key = (family, variant)
        if key not in self._models:
            raise RuntimeError(
                f"Modelo '{family}/{variant}' não encontrado. "
                f"Disponíveis para '{family}': {sorted(family_variants)}"
            )
        return self._models[key]


class OrangeMaskFilter:
    """
    Filtro simples em HSV para:
    1) prefilter global (pular inferência quando não há quase laranja no frame)
    2) validação por caixa (reduzir falso positivo)
    """

    def __init__(
        self,
        low1: Tuple[int, int, int] = (0, 80, 70),
        high1: Tuple[int, int, int] = (28, 255, 255),
        low2: Tuple[int, int, int] = (160, 80, 70),
        high2: Tuple[int, int, int] = (179, 255, 255),
        morph_open: int = 3,
        morph_close: int = 5,
    ):
        self.low1 = np.array(low1, dtype=np.uint8)
        self.high1 = np.array(high1, dtype=np.uint8)
        self.low2 = np.array(low2, dtype=np.uint8)
        self.high2 = np.array(high2, dtype=np.uint8)
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))

    def mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.low1, self.high1)
        m2 = cv2.inRange(hsv, self.low2, self.high2)
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close, iterations=1)
        return mask

    def global_ratio(self, bgr: np.ndarray, down_w: int = 240) -> float:
        h, w = bgr.shape[:2]
        if w <= 0 or h <= 0:
            return 0.0
        scale = down_w / float(w)
        down_h = max(1, int(round(h * scale)))
        small = cv2.resize(bgr, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
        m = self.mask(small)
        return float(np.count_nonzero(m)) / float(m.size + 1e-6)

    def box_ratio(self, bgr: np.ndarray, det: Detection) -> float:
        if det.area <= 0:
            return 0.0
        roi = bgr[det.y1 : det.y2, det.x1 : det.x2]
        if roi.size == 0:
            return 0.0
        m = self.mask(roi)
        return float(np.count_nonzero(m)) / float(m.size + 1e-6)


class YoloOnnxDetector:
    def __init__(
        self,
        model: ModelSpec,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        threads: int = 0,
        spinning: bool = False,
        graph_opt: str = "all",
    ):
        self.model = model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self._infer_lock = threading.Lock()

        sess_opt = ort.SessionOptions()
        if threads >= 0:
            sess_opt.intra_op_num_threads = int(threads)
        sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        graph_opt = graph_opt.lower()
        if graph_opt == "basic":
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif graph_opt == "extended":
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if not spinning:
            sess_opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
            sess_opt.add_session_config_entry("session.inter_op.allow_spinning", "0")

        self.session = ort.InferenceSession(
            str(model.path),
            sess_options=sess_opt,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        ih, iw = image.shape[:2]
        th, tw = self.model.input_h, self.model.input_w
        ratio = min(tw / float(iw), th / float(ih))
        nw, nh = int(round(iw * ratio)), int(round(ih * ratio))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        out = np.full((th, tw, 3), 114, dtype=np.uint8)
        pad_w = (tw - nw) // 2
        pad_h = (th - nh) // 2
        out[pad_h : pad_h + nh, pad_w : pad_w + nw] = resized
        return out, ratio, (pad_w, pad_h)

    def _preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        canvas, ratio, pad = self._letterbox(bgr)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return np.ascontiguousarray(blob), ratio, pad

    @staticmethod
    def _flatten_indices(indices: Sequence) -> List[int]:
        if indices is None:
            return []
        if len(indices) == 0:
            return []
        out: List[int] = []
        for v in indices:
            if isinstance(v, (list, tuple, np.ndarray)):
                out.append(int(v[0]))
            else:
                out.append(int(v))
        return out

    def detect(self, frame: np.ndarray, offset_xy: Tuple[int, int] = (0, 0)) -> List[Detection]:
        frame_h, frame_w = frame.shape[:2]
        inp, ratio, (pad_w, pad_h) = self._preprocess(frame)
        with self._infer_lock:
            raw = self.session.run(None, {self.input_name: inp})[0]

        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim != 2 or raw.shape[1] < 6:
            raise RuntimeError(f"Saída inesperada do modelo: shape={raw.shape}")

        boxes_xywh: List[List[int]] = []
        scores: List[float] = []
        prepared: List[Detection] = []

        for row in raw:
            conf = float(row[4])
            if conf < self.conf_thres:
                continue
            cls = int(row[5])
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])

            x1 = (x1 - pad_w) / max(ratio, 1e-6)
            y1 = (y1 - pad_h) / max(ratio, 1e-6)
            x2 = (x2 - pad_w) / max(ratio, 1e-6)
            y2 = (y2 - pad_h) / max(ratio, 1e-6)

            x1 = int(clamp(round(x1), 0, frame_w - 1))
            y1 = int(clamp(round(y1), 0, frame_h - 1))
            x2 = int(clamp(round(x2), 0, frame_w - 1))
            y2 = int(clamp(round(y2), 0, frame_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            det = Detection(
                x1=x1 + offset_xy[0],
                y1=y1 + offset_xy[1],
                x2=x2 + offset_xy[0],
                y2=y2 + offset_xy[1],
                conf=conf,
                cls=cls,
            )
            prepared.append(det)
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)

        if not prepared:
            return []

        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh,
            scores=scores,
            score_threshold=self.conf_thres,
            nms_threshold=self.iou_thres,
            top_k=self.max_det,
        )
        keep = self._flatten_indices(nms_indices)
        if not keep:
            return []

        kept = [prepared[i] for i in keep]
        kept.sort(key=lambda d: d.conf, reverse=True)
        return kept[: self.max_det]


@dataclass
class PipelineStats:
    total_frames: int = 0
    inferred_frames: int = 0
    skipped_stride: int = 0
    skipped_prefilter: int = 0
    total_infer_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    last_status: str = "init"

    def record(self, pipeline_ms: float, infer_ms: float) -> None:
        self.total_pipeline_ms += pipeline_ms
        self.total_infer_ms += infer_ms

    @property
    def avg_infer_ms(self) -> float:
        return (self.total_infer_ms / self.inferred_frames) if self.inferred_frames > 0 else 0.0

    @property
    def avg_pipeline_ms(self) -> float:
        return (self.total_pipeline_ms / self.total_frames) if self.total_frames > 0 else 0.0


@dataclass
class FollowConfig:
    """
    Configuração para gerar erro de seguimento de cone para controle de robô.

    Estratégia (IBVS prática):
    - erro angular: desvio do centro da caixa no eixo x (em rad, com modelo pinhole)
    - erro de distância: por métrica (se cone_height + intrínsecos) ou proxy por altura da caixa
    """

    camera_hfov_deg: float = 70.0
    camera_fx_px: float = 0.0
    camera_fy_px: float = 0.0
    cone_height_m: float = 0.45
    target_distance_m: float = 1.50
    target_box_height_ratio: float = 0.18
    ema_alpha: float = 0.45
    kp_ang: float = 1.80
    kd_ang: float = 0.15
    kp_dist: float = 0.90
    max_w: float = 1.30
    max_v: float = 0.70
    deadband_heading_deg: float = 1.00
    deadband_dist: float = 0.05
    v_slowdown_heading_deg: float = 25.0


@dataclass
class FollowState:
    found: bool = False
    status: str = "no_target"
    target: Optional[Detection] = None
    heading_error_rad: float = 0.0
    heading_error_deg: float = 0.0
    heading_rate_rad_s: float = 0.0
    lateral_error_norm: float = 0.0
    box_height_ratio: float = 0.0
    distance_proxy_error: float = 0.0
    distance_est_m: Optional[float] = None
    distance_error_m: Optional[float] = None
    distance_error_ctrl: float = 0.0
    tracking_quality: float = 0.0
    v_cmd: float = 0.0
    w_cmd: float = 0.0
    combined_error: float = 0.0


class ConeFollowErrorEstimator:
    def __init__(self, cfg: FollowConfig):
        self.cfg = cfg
        self._prev_t: Optional[float] = None
        self._prev_heading_ema: float = 0.0
        self._initialized = False
        self.frames = 0
        self.found_frames = 0
        self.sum_abs_heading_deg = 0.0
        self.sum_abs_dist_err = 0.0
        self.last_state = FollowState(found=False, status="init")

    def _fx_fy(self, frame_w: int) -> Tuple[float, float]:
        if self.cfg.camera_fx_px > 0:
            fx = self.cfg.camera_fx_px
        else:
            hfov_rad = math.radians(clamp(self.cfg.camera_hfov_deg, 1.0, 179.0))
            fx = (frame_w * 0.5) / max(math.tan(hfov_rad * 0.5), 1e-6)

        fy = self.cfg.camera_fy_px if self.cfg.camera_fy_px > 0 else fx
        return float(fx), float(fy)

    @staticmethod
    def _choose_target(detections: Sequence[Detection], frame_w: int, frame_h: int) -> Optional[Detection]:
        if not detections:
            return None

        best = None
        best_score = -1e9
        for d in detections:
            cx = 0.5 * (d.x1 + d.x2)
            center_err = abs((cx - 0.5 * frame_w) / max(1.0, 0.5 * frame_w))
            h_ratio = d.h / max(1.0, float(frame_h))
            score = (1.20 * d.conf) + (0.35 * h_ratio) - (0.55 * center_err)
            if score > best_score:
                best_score = score
                best = d
        return best

    def _dt(self, timestamp_s: float) -> float:
        if self._prev_t is None:
            self._prev_t = timestamp_s
            return 1.0 / 30.0
        dt = max(1e-3, timestamp_s - self._prev_t)
        self._prev_t = timestamp_s
        return dt

    def update(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        frame_index: int,
        timestamp_s: float,
    ) -> FollowState:
        del frame_index
        self.frames += 1
        dt = self._dt(timestamp_s)
        frame_h, frame_w = frame.shape[:2]

        target = self._choose_target(detections, frame_w=frame_w, frame_h=frame_h)
        if target is None:
            state = FollowState(found=False, status="no_target", tracking_quality=0.0, v_cmd=0.0, w_cmd=0.0)
            self.last_state = state
            return state

        self.found_frames += 1
        fx, fy = self._fx_fy(frame_w=frame_w)
        cx = 0.5 * (target.x1 + target.x2)
        px_offset = cx - (0.5 * frame_w)
        lateral_norm = px_offset / max(1.0, 0.5 * frame_w)
        heading_raw = math.atan2(px_offset, fx)

        if not self._initialized:
            heading_ema = heading_raw
            self._initialized = True
        else:
            alpha = clamp(self.cfg.ema_alpha, 0.01, 1.0)
            heading_ema = alpha * heading_raw + (1.0 - alpha) * self._prev_heading_ema

        heading_rate = (heading_ema - self._prev_heading_ema) / dt
        self._prev_heading_ema = heading_ema
        heading_deg = math.degrees(heading_ema)

        box_h_ratio = target.h / max(1.0, float(frame_h))
        distance_proxy_error = self.cfg.target_box_height_ratio - box_h_ratio

        distance_est_m: Optional[float] = None
        distance_error_m: Optional[float] = None
        if self.cfg.cone_height_m > 0.0 and target.h > 0:
            # Pinhole approximation: h_px ~= fy * H_real / Z  =>  Z ~= fy * H_real / h_px
            distance_est_m = (self.cfg.cone_height_m * fy) / float(target.h)
            distance_error_m = distance_est_m - self.cfg.target_distance_m

        dist_error_ctrl = distance_error_m if distance_error_m is not None else distance_proxy_error
        heading_ctrl = heading_ema

        if abs(heading_deg) < self.cfg.deadband_heading_deg:
            heading_ctrl = 0.0
        if abs(dist_error_ctrl) < self.cfg.deadband_dist:
            dist_error_ctrl = 0.0

        w_cmd = (self.cfg.kp_ang * heading_ctrl) + (self.cfg.kd_ang * heading_rate)
        w_cmd = clamp(w_cmd, -self.cfg.max_w, self.cfg.max_w)

        v_cmd = self.cfg.kp_dist * dist_error_ctrl
        slowdown = 1.0 - (abs(heading_deg) / max(1.0, self.cfg.v_slowdown_heading_deg))
        slowdown = clamp(slowdown, 0.20, 1.00)
        v_cmd *= slowdown
        v_cmd = clamp(v_cmd, -self.cfg.max_v, self.cfg.max_v)

        heading_norm = abs(heading_ctrl) / max(1e-3, math.radians(self.cfg.v_slowdown_heading_deg))
        if distance_error_m is not None:
            dist_den = max(0.10, self.cfg.target_distance_m)
        else:
            dist_den = max(0.01, self.cfg.target_box_height_ratio)
        dist_norm = abs(dist_error_ctrl) / dist_den
        combined_error = float(math.sqrt((heading_norm * heading_norm) + (dist_norm * dist_norm)))

        track_q = target.conf * (1.0 - 0.25 * abs(lateral_norm))
        track_q = float(clamp(track_q, 0.0, 1.0))

        self.sum_abs_heading_deg += abs(heading_deg)
        self.sum_abs_dist_err += abs(dist_error_ctrl)

        state = FollowState(
            found=True,
            status="tracking",
            target=target,
            heading_error_rad=float(heading_ema),
            heading_error_deg=float(heading_deg),
            heading_rate_rad_s=float(heading_rate),
            lateral_error_norm=float(lateral_norm),
            box_height_ratio=float(box_h_ratio),
            distance_proxy_error=float(distance_proxy_error),
            distance_est_m=(float(distance_est_m) if distance_est_m is not None else None),
            distance_error_m=(float(distance_error_m) if distance_error_m is not None else None),
            distance_error_ctrl=float(dist_error_ctrl),
            tracking_quality=track_q,
            v_cmd=float(v_cmd),
            w_cmd=float(w_cmd),
            combined_error=combined_error,
        )
        self.last_state = state
        return state

    def summary(self) -> Dict[str, float]:
        frames = max(1, self.frames)
        found = max(1, self.found_frames)
        return {
            "frames": float(self.frames),
            "found_frames": float(self.found_frames),
            "found_ratio": float(self.found_frames / frames),
            "mean_abs_heading_deg": float(self.sum_abs_heading_deg / found),
            "mean_abs_distance_error_ctrl": float(self.sum_abs_dist_err / found),
        }


def follow_state_to_record(frame_index: int, timestamp_s: float, state: FollowState) -> Dict[str, object]:
    rec: Dict[str, object] = {
        "frame": int(frame_index),
        "t_sec": float(timestamp_s),
        "found": bool(state.found),
        "status": state.status,
        "heading_error_rad": float(state.heading_error_rad),
        "heading_error_deg": float(state.heading_error_deg),
        "heading_rate_rad_s": float(state.heading_rate_rad_s),
        "lateral_error_norm": float(state.lateral_error_norm),
        "box_height_ratio": float(state.box_height_ratio),
        "distance_proxy_error": float(state.distance_proxy_error),
        "distance_error_ctrl": float(state.distance_error_ctrl),
        "v_cmd": float(state.v_cmd),
        "w_cmd": float(state.w_cmd),
        "combined_error": float(state.combined_error),
        "tracking_quality": float(state.tracking_quality),
    }

    if state.distance_est_m is not None:
        rec["distance_est_m"] = float(state.distance_est_m)
    if state.distance_error_m is not None:
        rec["distance_error_m"] = float(state.distance_error_m)
    if state.target is not None:
        rec["target_conf"] = float(state.target.conf)
        rec["target_bbox"] = [int(state.target.x1), int(state.target.y1), int(state.target.x2), int(state.target.y2)]

    return rec


def detection_to_record(det: Detection) -> Dict[str, object]:
    return {
        "bbox": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
        "conf": float(det.conf),
        "cls": int(det.cls),
        "orange_ratio": float(det.orange_ratio),
        "width": int(det.w),
        "height": int(det.h),
        "area": int(det.area),
    }


def pipeline_stats_to_record(pipeline: "ConePipeline") -> Dict[str, object]:
    return {
        "total_frames": int(pipeline.stats.total_frames),
        "inferred_frames": int(pipeline.stats.inferred_frames),
        "skipped_stride": int(pipeline.stats.skipped_stride),
        "skipped_prefilter": int(pipeline.stats.skipped_prefilter),
        "avg_infer_ms": float(pipeline.stats.avg_infer_ms),
        "avg_pipeline_ms": float(pipeline.stats.avg_pipeline_ms),
        "last_status": pipeline.stats.last_status,
    }


def encode_image(image: np.ndarray, ext: str = ".jpg", jpeg_quality: int = 90) -> bytes:
    params: List[int] = []
    if ext.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(clamp(jpeg_quality, 1, 100))]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Falha ao codificar imagem como {ext}")
    return bytes(buf.tobytes())


def decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Falha ao decodificar imagem enviada para a API.")
    return frame


class ConePipeline:
    def __init__(
        self,
        detector: YoloOnnxDetector,
        orange_filter: OrangeMaskFilter,
        det_every: int = 1,
        prefilter_enabled: bool = True,
        prefilter_min_ratio: float = 0.003,
        min_box_orange_ratio: float = 0.02,
        use_roi: bool = False,
        roi_norm: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        hold_last_on_stride: bool = True,
    ):
        self.detector = detector
        self.orange_filter = orange_filter
        self.det_every = max(1, det_every)
        self.prefilter_enabled = prefilter_enabled
        self.prefilter_min_ratio = prefilter_min_ratio
        self.min_box_orange_ratio = min_box_orange_ratio
        self.use_roi = use_roi
        self.roi_norm = roi_norm
        self.hold_last_on_stride = hold_last_on_stride
        self.stats = PipelineStats()
        self.last_detections: List[Detection] = []

    def _resolve_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        if not self.use_roi:
            return 0, 0, w, h
        x1n, y1n, x2n, y2n = self.roi_norm
        x1 = int(clamp(round(x1n * w), 0, w - 1))
        y1 = int(clamp(round(y1n * h), 0, h - 1))
        x2 = int(clamp(round(x2n * w), 1, w))
        y2 = int(clamp(round(y2n * h), 1, h))
        if x2 <= x1 or y2 <= y1:
            return 0, 0, w, h
        return x1, y1, x2, y2

    def run(self, frame: np.ndarray, frame_index: int) -> List[Detection]:
        t0 = time.perf_counter()
        self.stats.total_frames += 1

        if frame_index % self.det_every != 0:
            self.stats.skipped_stride += 1
            self.stats.last_status = "stride_skip"
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.stats.record(pipeline_ms=elapsed, infer_ms=0.0)
            return self.last_detections if self.hold_last_on_stride else []

        x1, y1, x2, y2 = self._resolve_roi(frame)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            self.stats.last_status = "empty_roi"
            self.last_detections = []
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.stats.record(pipeline_ms=elapsed, infer_ms=0.0)
            return []

        if self.prefilter_enabled:
            ratio = self.orange_filter.global_ratio(roi)
            if ratio < self.prefilter_min_ratio:
                self.stats.skipped_prefilter += 1
                self.stats.last_status = f"prefilter_skip({ratio:.4f})"
                self.last_detections = []
                elapsed = (time.perf_counter() - t0) * 1000.0
                self.stats.record(pipeline_ms=elapsed, infer_ms=0.0)
                return []

        inf_t0 = time.perf_counter()
        detections = self.detector.detect(roi, offset_xy=(x1, y1))
        infer_ms = (time.perf_counter() - inf_t0) * 1000.0
        self.stats.inferred_frames += 1

        if self.min_box_orange_ratio > 0:
            accepted: List[Detection] = []
            for d in detections:
                r = self.orange_filter.box_ratio(frame, d)
                d.orange_ratio = r
                if r >= self.min_box_orange_ratio:
                    accepted.append(d)
            detections = accepted

        self.last_detections = detections
        self.stats.last_status = "inferred"
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.stats.record(pipeline_ms=elapsed, infer_ms=infer_ms)
        return detections


def _alpha_rect(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    frame_h, frame_w = image.shape[:2]
    x1 = int(clamp(x1, 0, frame_w - 1))
    y1 = int(clamp(y1, 0, frame_h - 1))
    x2 = int(clamp(x2, 0, frame_w - 1))
    y2 = int(clamp(y2, 0, frame_h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, clamp(alpha, 0.0, 1.0), image, 1.0 - clamp(alpha, 0.0, 1.0), 0.0, dst=image)


def _draw_panel(
    image: np.ndarray,
    x: int,
    y: int,
    lines: Sequence[str],
    bg_color: Tuple[int, int, int] = (20, 20, 20),
    fg_color: Tuple[int, int, int] = (240, 240, 240),
    alpha: float = 0.55,
) -> Tuple[int, int, int, int]:
    if not lines:
        return x, y, x, y

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.50
    thickness = 1
    pad = 8
    line_gap = 6

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_w = max(w for w, _ in text_sizes)
    max_h = max(h for _, h in text_sizes)
    line_h = max_h + line_gap

    panel_w = max_w + (2 * pad)
    panel_h = (line_h * len(lines)) + (2 * pad) - line_gap
    frame_h, frame_w = image.shape[:2]

    x1 = int(clamp(x, 0, max(0, frame_w - panel_w - 1)))
    y1 = int(clamp(y, 0, max(0, frame_h - panel_h - 1)))
    x2 = min(frame_w - 1, x1 + panel_w)
    y2 = min(frame_h - 1, y1 + panel_h)

    _alpha_rect(image, x1, y1, x2, y2, bg_color, alpha)
    cv2.rectangle(image, (x1, y1), (x2, y2), (120, 120, 120), 1, cv2.LINE_AA)

    yy = y1 + pad + max_h
    for line in lines:
        cv2.putText(image, line, (x1 + pad, yy), font, font_scale, fg_color, thickness, cv2.LINE_AA)
        yy += line_h

    return x1, y1, x2, y2


def _draw_signed_bar(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    max_value: float,
    label: str,
    neg_color: Tuple[int, int, int],
    pos_color: Tuple[int, int, int],
) -> None:
    if w <= 8 or h <= 8:
        return

    _alpha_rect(image, x, y, x + w, y + h, (16, 16, 16), 0.50)
    cv2.rectangle(image, (x, y), (x + w, y + h), (120, 120, 120), 1, cv2.LINE_AA)

    center_x = x + (w // 2)
    cv2.line(image, (center_x, y + 2), (center_x, y + h - 2), (150, 150, 150), 1, cv2.LINE_AA)

    ratio = clamp(value / max(max_value, 1e-6), -1.0, 1.0)
    half = max(1, (w // 2) - 3)
    if ratio >= 0:
        bx1 = center_x
        bx2 = center_x + int(round(ratio * half))
        color = pos_color
    else:
        bx1 = center_x + int(round(ratio * half))
        bx2 = center_x
        color = neg_color

    if bx2 > bx1:
        cv2.rectangle(image, (bx1, y + 2), (bx2, y + h - 2), color, -1)

    cv2.putText(
        image,
        f"{label} {value:+.2f}",
        (x + 6, y + h - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (236, 236, 236),
        1,
        cv2.LINE_AA,
    )


def _detection_color(conf: float) -> Tuple[int, int, int]:
    if conf >= 0.70:
        return 60, 220, 70
    if conf >= 0.45:
        return 40, 200, 240
    return 80, 150, 255


def draw_overlay(
    frame: np.ndarray,
    detections: Sequence[Detection],
    model_tag: str,
    status: str,
    avg_infer_ms: float,
    show_orange_ratio: bool,
    follow_state: Optional[FollowState] = None,
    avg_pipeline_ms: float = 0.0,
    overlay_ctx: Optional[OverlayContext] = None,
    frame_index: int = 0,
) -> np.ndarray:
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]
    center_x, center_y = frame_w // 2, frame_h // 2

    # Center reticle helps camera alignment and manual tuning.
    cv2.circle(out, (center_x, center_y), 8, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.line(out, (center_x - 18, center_y), (center_x + 18, center_y), (140, 140, 140), 1, cv2.LINE_AA)
    cv2.line(out, (center_x, center_y - 18), (center_x, center_y + 18), (140, 140, 140), 1, cv2.LINE_AA)

    if overlay_ctx is not None and overlay_ctx.use_roi:
        x1n, y1n, x2n, y2n = overlay_ctx.roi_norm
        rx1 = int(clamp(round(x1n * frame_w), 0, frame_w - 1))
        ry1 = int(clamp(round(y1n * frame_h), 0, frame_h - 1))
        rx2 = int(clamp(round(x2n * frame_w), 1, frame_w - 1))
        ry2 = int(clamp(round(y2n * frame_h), 1, frame_h - 1))
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (245, 210, 90), 2, cv2.LINE_AA)
        _alpha_rect(out, rx1, max(0, ry1 - 24), min(frame_w - 1, rx1 + 120), ry1, (40, 60, 95), 0.65)
        cv2.putText(out, "ROI active", (rx1 + 6, max(14, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 240, 210), 1, cv2.LINE_AA)

    for idx, d in enumerate(detections):
        color = _detection_color(d.conf)
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2, cv2.LINE_AA)

        lbl = f"cone#{idx + 1} {d.conf:.2f}"
        if show_orange_ratio and d.orange_ratio >= 0:
            lbl += f" o:{d.orange_ratio:.2f}"

        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
        label_x1 = int(clamp(d.x1, 0, frame_w - 1))
        label_y2 = d.y1 - 4
        if label_y2 < (th + 8):
            label_y2 = min(frame_h - 2, d.y1 + th + 10)
        label_y1 = int(clamp(label_y2 - th - 10, 0, frame_h - 1))
        label_x2 = int(clamp(label_x1 + tw + 12, 0, frame_w - 1))

        _alpha_rect(out, label_x1, label_y1, label_x2, label_y2, color, 0.58)
        cv2.putText(
            out,
            lbl,
            (label_x1 + 6, label_y2 - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.48,
            (12, 12, 12),
            1,
            cv2.LINE_AA,
        )

    status_color = (70, 225, 110)
    if "skip" in status:
        status_color = (50, 205, 245)
    elif "empty" in status:
        status_color = (80, 155, 255)

    info_lines = [
        f"model {model_tag}",
        f"frame {frame_index} | dets {len(detections)} | status {status}",
        f"infer {avg_infer_ms:.1f} ms | pipeline {avg_pipeline_ms:.1f} ms",
    ]
    if overlay_ctx is not None:
        rt = overlay_ctx.runtime
        info_lines.append(
            f"profile {overlay_ctx.profile} | conf {rt.conf:.2f} | iou {rt.iou:.2f} | det_every {rt.det_every}"
        )
        info_lines.append(
            f"prefilter {'on' if rt.prefilter_enabled else 'off'} | global {rt.prefilter_min_ratio:.4f} | box {rt.min_box_orange_ratio:.3f}"
        )
    _draw_panel(
        out,
        x=10,
        y=10,
        lines=info_lines,
        bg_color=(22, 34, 48),
        fg_color=status_color,
        alpha=0.58,
    )

    if follow_state is not None:
        if follow_state.target is not None:
            tx = int((follow_state.target.x1 + follow_state.target.x2) * 0.5)
            ty = int((follow_state.target.y1 + follow_state.target.y2) * 0.5)
            cv2.line(out, (center_x, center_y), (tx, ty), (225, 225, 225), 1, cv2.LINE_AA)
            cv2.circle(out, (tx, ty), 9, (255, 255, 255), 2, cv2.LINE_AA)

        follow_lines = [
            f"follow {follow_state.status} | q {follow_state.tracking_quality:.2f} | found {int(follow_state.found)}",
            f"yaw {follow_state.heading_error_deg:+.2f} deg | dist_err {follow_state.distance_error_ctrl:+.3f}",
            f"cmd v {follow_state.v_cmd:+.2f} | w {follow_state.w_cmd:+.2f} | err {follow_state.combined_error:.2f}",
        ]
        fx1, fy1, fx2, fy2 = _draw_panel(
            out,
            x=max(10, frame_w - 470),
            y=10,
            lines=follow_lines,
            bg_color=(40, 36, 18),
            fg_color=(255, 235, 165),
            alpha=0.62,
        )

        bar_w = max(120, min(430, frame_w - fx1 - 14))
        bar_h = 22
        bar_x = fx1 + 6
        bar_y = fy2 + 8
        _draw_signed_bar(
            out,
            x=bar_x,
            y=bar_y,
            w=bar_w,
            h=bar_h,
            value=follow_state.heading_error_deg,
            max_value=25.0,
            label="yaw_deg",
            neg_color=(80, 130, 255),
            pos_color=(80, 230, 150),
        )
        _draw_signed_bar(
            out,
            x=bar_x,
            y=bar_y + bar_h + 8,
            w=bar_w,
            h=bar_h,
            value=follow_state.distance_error_ctrl,
            max_value=1.0,
            label="dist_err",
            neg_color=(100, 185, 255),
            pos_color=(120, 225, 150),
        )

    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def parse_roi(roi_text: str) -> Tuple[float, float, float, float]:
    parts = [float(v.strip()) for v in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI deve ter 4 valores: x1,y1,x2,y2 (normalizados em [0,1])")
    x1, y1, x2, y2 = parts
    for v in parts:
        if v < 0.0 or v > 1.0:
            raise ValueError("ROI normalizado deve usar intervalo [0,1]")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI inválido: esperado x2>x1 e y2>y1")
    return x1, y1, x2, y2


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detecção de cones em imagem/vídeo com modelos YOLO ONNX da v4 + OpenCV."
    )
    p.add_argument("--source", default="", help="Caminho da imagem/vídeo/pasta, ou índice da câmera (ex: 0).")
    p.add_argument("--output", default="", help="Arquivo de saída (imagem/vídeo) ou pasta de saída para diretório.")
    p.add_argument("--api", default="", help="Inicia servidor HTTP no formato host:port (ex: localhost:9820).")
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT), help="Raiz dos modelos (padrão: v4/yolov26n).")
    p.add_argument("--list-models", action="store_true", help="Lista os modelos detectados e sai.")

    p.add_argument("--profile", choices=sorted(PRESETS.keys()), default="balanced")
    p.add_argument("--family", default="auto", help="aug, noaug, ou auto")
    p.add_argument("--variant", default="auto", help="simplified, normal, ou auto")

    p.add_argument("--conf", type=float, default=None, help="Threshold de confiança.")
    p.add_argument("--iou", type=float, default=None, help="Threshold IoU do NMS.")
    p.add_argument("--max-det", type=int, default=None, help="Número máximo de detecções por frame.")
    p.add_argument("--det-every", type=int, default=None, help="Roda inferência a cada N frames.")

    p.add_argument("--threads", type=int, default=None, help="intra_op threads do ONNX Runtime (0 = auto).")
    p.add_argument(
        "--graph-opt",
        choices=["basic", "extended", "all"],
        default="all",
        help="Nível de otimização de grafo no ONNX Runtime.",
    )
    p.add_argument("--enable-spinning", action="store_true", help="Habilita thread spinning no ONNX Runtime.")

    p.add_argument("--prefilter", action="store_true", help="Força prefilter HSV ligado.")
    p.add_argument("--no-prefilter", action="store_true", help="Força prefilter HSV desligado.")
    p.add_argument("--prefilter-min-ratio", type=float, default=None, help="Razão mínima global de laranja para inferir.")
    p.add_argument("--min-box-orange-ratio", type=float, default=None, help="Razão mínima de laranja por caixa.")

    p.add_argument("--roi", default="", help="ROI normalizado x1,y1,x2,y2 (ex: 0.0,0.35,1.0,1.0).")

    p.add_argument("--camera-width", type=int, default=0, help="Largura solicitada para câmera.")
    p.add_argument("--camera-height", type=int, default=0, help="Altura solicitada para câmera.")
    p.add_argument("--buffer-size", type=int, default=1, help="cv.CAP_PROP_BUFFERSIZE para captura.")
    p.add_argument("--drop-grabs", type=int, default=0, help="Quantidade de cap.grab() extra por loop.")

    p.add_argument("--show", action="store_true", help="Mostra janela com preview.")
    p.add_argument("--show-orange-ratio", action="store_true", help="Mostra razão de laranja na label da caixa.")
    p.add_argument("--save-txt", default="", help="Salva log simples de detecções em txt.")

    p.add_argument("--follow", action="store_true", help="Gera erro de seguimento para robô (yaw/distância + v/w).")
    p.add_argument("--follow-jsonl", default="", help="Salva telemetria de seguimento por frame em JSONL.")
    p.add_argument("--camera-hfov-deg", type=float, default=70.0, help="HFOV da câmera em graus (se não passar fx).")
    p.add_argument("--camera-fx-px", type=float, default=0.0, help="Focal fx em pixels (opcional, sobrepõe HFOV).")
    p.add_argument("--camera-fy-px", type=float, default=0.0, help="Focal fy em pixels (opcional).")
    p.add_argument("--cone-height-m", type=float, default=0.45, help="Altura real do cone (metros).")
    p.add_argument("--target-distance-m", type=float, default=1.50, help="Distância alvo ao cone (metros).")
    p.add_argument(
        "--target-box-height-ratio",
        type=float,
        default=0.18,
        help="Altura alvo da bbox / altura da imagem (fallback quando não há distância métrica).",
    )
    p.add_argument("--error-ema-alpha", type=float, default=0.45, help="EMA para suavizar erro angular.")
    p.add_argument("--follow-kp-ang", type=float, default=1.80, help="Ganho P angular (w).")
    p.add_argument("--follow-kd-ang", type=float, default=0.15, help="Ganho D angular (w).")
    p.add_argument("--follow-kp-dist", type=float, default=0.90, help="Ganho P linear (v).")
    p.add_argument("--follow-max-w", type=float, default=1.30, help="Saturação de velocidade angular (rad/s).")
    p.add_argument("--follow-max-v", type=float, default=0.70, help="Saturação de velocidade linear (m/s ou proxy).")
    p.add_argument("--follow-deadband-deg", type=float, default=1.0, help="Deadband angular em graus.")
    p.add_argument("--follow-deadband-dist", type=float, default=0.05, help="Deadband de distância.")
    p.add_argument(
        "--follow-v-slowdown-deg",
        type=float,
        default=25.0,
        help="Reduz v conforme erro angular absoluto cresce.",
    )

    return p


def resolve_runtime(args: argparse.Namespace) -> RuntimePreset:
    base = PRESETS[args.profile]
    family = args.family if args.family != "auto" else base.family
    variant = args.variant if args.variant != "auto" else base.variant

    prefilter_enabled = base.prefilter_enabled
    if args.prefilter:
        prefilter_enabled = True
    if args.no_prefilter:
        prefilter_enabled = False

    return RuntimePreset(
        family=family,
        variant=variant,
        conf=args.conf if args.conf is not None else base.conf,
        iou=args.iou if args.iou is not None else base.iou,
        max_det=args.max_det if args.max_det is not None else base.max_det,
        det_every=args.det_every if args.det_every is not None else base.det_every,
        prefilter_enabled=prefilter_enabled,
        prefilter_min_ratio=(
            args.prefilter_min_ratio if args.prefilter_min_ratio is not None else base.prefilter_min_ratio
        ),
        min_box_orange_ratio=(
            args.min_box_orange_ratio if args.min_box_orange_ratio is not None else base.min_box_orange_ratio
        ),
        threads=args.threads if args.threads is not None else base.threads,
    )


def write_txt_log(path: Path, lines: Iterable[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def run_frame_job(
    frame: np.ndarray,
    frame_index: int,
    timestamp_s: float,
    pipeline: ConePipeline,
    model_tag: str,
    overlay_ctx: OverlayContext,
    show_orange_ratio: bool,
    follow_estimator: Optional[ConeFollowErrorEstimator] = None,
    follow_log_file=None,
    follow_log_extra: Optional[Dict[str, object]] = None,
) -> FrameProcessResult:
    detections = pipeline.run(frame, frame_index=frame_index)
    follow_state: Optional[FollowState] = None
    if follow_estimator is not None:
        follow_state = follow_estimator.update(
            frame=frame,
            detections=detections,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )
        if follow_log_file is not None:
            rec = follow_state_to_record(frame_index=frame_index, timestamp_s=timestamp_s, state=follow_state)
            if follow_log_extra:
                rec.update(follow_log_extra)
            follow_log_file.write(json.dumps(rec, ensure_ascii=True) + "\n")

    rendered = draw_overlay(
        frame=frame,
        detections=detections,
        model_tag=model_tag,
        status=pipeline.stats.last_status,
        avg_infer_ms=pipeline.stats.avg_infer_ms,
        avg_pipeline_ms=pipeline.stats.avg_pipeline_ms,
        show_orange_ratio=show_orange_ratio,
        overlay_ctx=overlay_ctx,
        frame_index=frame_index,
        follow_state=follow_state,
    )
    return FrameProcessResult(
        detections=detections,
        rendered=rendered,
        follow_state=follow_state,
        status=pipeline.stats.last_status,
        avg_infer_ms=pipeline.stats.avg_infer_ms,
        avg_pipeline_ms=pipeline.stats.avg_pipeline_ms,
    )


def open_capture_source(
    source: str,
    camera_width: int,
    camera_height: int,
    buffer_size: int,
) -> Tuple[cv2.VideoCapture, bool, float]:
    source_is_cam = False
    cap_source: object = source

    if source.isdigit() and not Path(source).exists():
        source_is_cam = True
        cap_source = int(source)

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Falha ao abrir source: {source}")

    if source_is_cam:
        if camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        if camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    if buffer_size > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or fps_in > 500:
        fps_in = 30.0

    return cap, source_is_cam, float(fps_in)


def finalize_video_run(
    start_t: float,
    frame_count: int,
    pipeline: ConePipeline,
    follow_estimator: Optional[ConeFollowErrorEstimator],
) -> VideoRunResult:
    elapsed = time.perf_counter() - start_t
    eff_fps = frame_count / elapsed if elapsed > 0 else 0.0

    logs: List[str] = []
    logs.append(f"frames={frame_count}")
    logs.append(f"elapsed_s={elapsed:.2f}")
    logs.append(f"effective_fps={eff_fps:.2f}")
    logs.append(f"inferred_frames={pipeline.stats.inferred_frames}")
    logs.append(f"skip_stride={pipeline.stats.skipped_stride}")
    logs.append(f"skip_prefilter={pipeline.stats.skipped_prefilter}")
    logs.append(f"avg_infer_ms={pipeline.stats.avg_infer_ms:.2f}")
    logs.append(f"avg_pipeline_ms={pipeline.stats.avg_pipeline_ms:.2f}")

    follow_summary = None
    if follow_estimator is not None:
        follow_summary = follow_estimator.summary()
        logs.append(f"follow_found_ratio={follow_summary['found_ratio']:.3f}")
        logs.append(f"follow_mean_abs_heading_deg={follow_summary['mean_abs_heading_deg']:.3f}")
        logs.append(f"follow_mean_abs_distance_error={follow_summary['mean_abs_distance_error_ctrl']:.3f}")

    return VideoRunResult(
        logs=logs,
        frame_count=frame_count,
        elapsed_s=float(elapsed),
        effective_fps=float(eff_fps),
        follow_summary=follow_summary,
    )


def process_image_file(
    image_path: Path,
    out_path: Path,
    pipeline: ConePipeline,
    model_tag: str,
    overlay_ctx: OverlayContext,
    show: bool,
    show_orange_ratio: bool,
    follow_estimator: Optional[ConeFollowErrorEstimator] = None,
    follow_log_file=None,
) -> Tuple[int, str]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Falha ao abrir imagem: {image_path}")

    result = run_frame_job(
        frame=frame,
        frame_index=0,
        timestamp_s=0.0,
        pipeline=pipeline,
        model_tag=model_tag,
        overlay_ctx=overlay_ctx,
        show_orange_ratio=show_orange_ratio,
        follow_estimator=follow_estimator,
        follow_log_file=follow_log_file,
    )
    ensure_parent(out_path)
    cv2.imwrite(str(out_path), result.rendered)

    if show:
        cv2.imshow("cone_detector_v4", result.rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    msg = f"{image_path.name}: {len(result.detections)} cones"
    if result.follow_state is not None:
        msg += (
            " | "
            f"yaw={result.follow_state.heading_error_deg:+.2f}deg "
            f"dist_err={result.follow_state.distance_error_ctrl:+.3f} "
            f"v={result.follow_state.v_cmd:+.2f} "
            f"w={result.follow_state.w_cmd:+.2f}"
        )
    return len(result.detections), msg


def process_image_dir(
    in_dir: Path,
    out_dir: Path,
    pipeline: ConePipeline,
    model_tag: str,
    overlay_ctx: OverlayContext,
    show_orange_ratio: bool,
    follow_estimator: Optional[ConeFollowErrorEstimator] = None,
    follow_log_file=None,
) -> Tuple[List[str], int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs: List[str] = []
    total = 0

    images = [p for p in sorted(in_dir.iterdir()) if p.is_file() and is_image_path(p)]
    if not images:
        raise RuntimeError(f"Nenhuma imagem encontrada em: {in_dir}")

    for idx, image_path in enumerate(images):
        frame = cv2.imread(str(image_path))
        if frame is None:
            logs.append(f"{image_path.name}: erro ao abrir")
            continue
        result = run_frame_job(
            frame=frame,
            frame_index=idx,
            timestamp_s=float(idx),
            pipeline=pipeline,
            model_tag=model_tag,
            overlay_ctx=overlay_ctx,
            show_orange_ratio=show_orange_ratio,
            follow_estimator=follow_estimator,
            follow_log_file=follow_log_file,
            follow_log_extra={"image_name": image_path.name},
        )
        total += len(result.detections)
        out_path = out_dir / image_path.name
        cv2.imwrite(str(out_path), result.rendered)
        msg = f"{image_path.name}: {len(result.detections)} cones"
        if result.follow_state is not None:
            msg += (
                " | "
                f"yaw={result.follow_state.heading_error_deg:+.2f}deg "
                f"dist_err={result.follow_state.distance_error_ctrl:+.3f} "
                f"v={result.follow_state.v_cmd:+.2f} "
                f"w={result.follow_state.w_cmd:+.2f}"
            )
        logs.append(msg)

    return logs, total


def process_video_or_camera(
    source: str,
    out_path: Optional[Path],
    pipeline: ConePipeline,
    model_tag: str,
    overlay_ctx: OverlayContext,
    show: bool,
    show_orange_ratio: bool,
    camera_width: int,
    camera_height: int,
    buffer_size: int,
    drop_grabs: int,
    follow_estimator: Optional[ConeFollowErrorEstimator] = None,
    follow_log_file=None,
    max_frames: Optional[int] = None,
) -> VideoRunResult:
    cap, _source_is_cam, fps_in = open_capture_source(
        source=source,
        camera_width=camera_width,
        camera_height=camera_height,
        buffer_size=buffer_size,
    )
    writer = None
    frame_idx = 0
    start_t = time.perf_counter()
    frame_count = 0

    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            for _ in range(max(0, drop_grabs)):
                if not cap.grab():
                    break

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp_s = frame_idx / max(1e-6, fps_in)
            result = run_frame_job(
                frame=frame,
                frame_index=frame_idx,
                timestamp_s=timestamp_s,
                pipeline=pipeline,
                model_tag=model_tag,
                overlay_ctx=overlay_ctx,
                show_orange_ratio=show_orange_ratio,
                follow_estimator=follow_estimator,
                follow_log_file=follow_log_file,
            )

            if out_path is not None:
                if writer is None:
                    ensure_parent(out_path)
                    h, w = result.rendered.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (w, h))
                writer.write(result.rendered)

            if show:
                cv2.imshow("cone_detector_v4", result.rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            frame_count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return finalize_video_run(
        start_t=start_t,
        frame_count=frame_count,
        pipeline=pipeline,
        follow_estimator=follow_estimator,
    )


def video_run_result_to_record(
    result: VideoRunResult,
    pipeline: ConePipeline,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    rec: Dict[str, object] = {
        "frames": int(result.frame_count),
        "elapsed_s": float(result.elapsed_s),
        "effective_fps": float(result.effective_fps),
        "logs": list(result.logs),
        "pipeline": pipeline_stats_to_record(pipeline),
    }
    if output_path is not None:
        rec["output_video"] = str(output_path)
    if result.follow_summary is not None:
        rec["follow_summary"] = result.follow_summary
    return rec


def parse_api_bind(bind_addr: str) -> Tuple[str, int]:
    if ":" not in bind_addr:
        raise ValueError("Use --api no formato host:port, por exemplo localhost:9820")
    host, port_text = bind_addr.rsplit(":", 1)
    host = host.strip() or "0.0.0.0"
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"Porta inválida em --api: {port_text}") from exc
    if port < 0 or port > 65535:
        raise ValueError(f"Porta fora do intervalo válido: {port}")
    return host, port


def resolve_client_filesystem_path(path_text: str, client_cwd: Optional[str] = None) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    if client_cwd:
        return (Path(client_cwd).expanduser() / path).resolve()
    return path


def resolve_client_source_ref(source_text: str, client_cwd: Optional[str] = None) -> str:
    source_text = str(source_text).strip()
    if not source_text:
        return source_text
    if source_text.isdigit():
        return source_text
    if "://" in source_text:
        return source_text
    return str(resolve_client_filesystem_path(source_text, client_cwd=client_cwd))


def normalize_image_output_format(output_format: Optional[str]) -> str:
    fmt = (output_format or "jpg").strip().lower().lstrip(".")
    if fmt == "jpeg":
        fmt = "jpg"
    ext = f".{fmt}"
    if ext not in IMAGE_EXTS:
        raise ValueError(f"Formato de imagem inválido para output: {output_format}")
    return ext


def resolve_requested_image_output(
    requested: Optional[str],
    client_cwd: Optional[str] = None,
    output_format: Optional[str] = None,
) -> Optional[Path]:
    if not requested:
        return None
    output_path = resolve_client_filesystem_path(str(requested), client_cwd=client_cwd)
    if output_path.suffix:
        return output_path
    return output_path.with_suffix(normalize_image_output_format(output_format))


class ApiStreamWorker:
    def __init__(
        self,
        app_ctx: AppContext,
        stream_id: str,
        source: str,
        output_path: Optional[Path],
        max_frames: Optional[int],
        max_seconds: Optional[float],
    ):
        self.app_ctx = app_ctx
        self.state = StreamJobState(
            stream_id=stream_id,
            source=source,
            output_path=output_path,
            max_frames=max_frames,
            max_seconds=max_seconds,
            started_at=time.time(),
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"v4-stream-{stream_id[:8]}")

    def start(self) -> None:
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> Dict[str, object]:
        self._stop_event.set()
        self._thread.join(timeout=join_timeout)
        payload = self.snapshot()
        payload["stopped"] = True
        return payload

    def latest_frame(self) -> bytes:
        with self._lock:
            if self.state.latest_frame_jpeg is None:
                raise RuntimeError(f"Stream '{self.state.stream_id}' ainda não produziu frame renderizado.")
            return self.state.latest_frame_jpeg

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            payload: Dict[str, object] = {
                "ok": self.state.error is None,
                "stream_id": self.state.stream_id,
                "source": self.state.source,
                "running": bool(self.state.running),
                "frames_processed": int(self.state.frames_processed),
                "last_timestamp_s": float(self.state.last_timestamp_s),
                "last_status": self.state.last_status,
                "last_detections": list(self.state.last_detections),
                "output_video": (str(self.state.output_path) if self.state.output_path is not None else ""),
                "started_at": float(self.state.started_at),
                "ended_at": (float(self.state.ended_at) if self.state.ended_at is not None else None),
                "max_frames": self.state.max_frames,
                "max_seconds": self.state.max_seconds,
            }
            if self.state.last_follow is not None:
                payload["last_follow"] = self.state.last_follow
            if self.state.summary is not None:
                payload["summary"] = self.state.summary
            if self.state.error is not None:
                payload["error"] = self.state.error
            return payload

    def _run(self) -> None:
        pipeline = self.app_ctx.create_pipeline()
        follow_estimator = self.app_ctx.create_follow_estimator()
        writer = None
        cap = None
        frame_idx = 0
        start_t = time.perf_counter()

        try:
            cap, _source_is_cam, fps_in = open_capture_source(
                source=self.state.source,
                camera_width=self.app_ctx.camera_width,
                camera_height=self.app_ctx.camera_height,
                buffer_size=self.app_ctx.buffer_size,
            )

            while not self._stop_event.is_set():
                if self.state.max_frames is not None and frame_idx >= self.state.max_frames:
                    break
                if self.state.max_seconds is not None and (time.perf_counter() - start_t) >= self.state.max_seconds:
                    break

                for _ in range(max(0, self.app_ctx.drop_grabs)):
                    if not cap.grab():
                        break

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                timestamp_s = frame_idx / max(1e-6, fps_in)
                result = run_frame_job(
                    frame=frame,
                    frame_index=frame_idx,
                    timestamp_s=timestamp_s,
                    pipeline=pipeline,
                    model_tag=self.app_ctx.model_tag,
                    overlay_ctx=self.app_ctx.overlay_ctx,
                    show_orange_ratio=self.app_ctx.show_orange_ratio,
                    follow_estimator=follow_estimator,
                )

                if self.state.output_path is not None:
                    if writer is None:
                        ensure_parent(self.state.output_path)
                        h, w = result.rendered.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(self.state.output_path), fourcc, fps_in, (w, h))
                    writer.write(result.rendered)

                follow_payload = (
                    follow_state_to_record(frame_index=frame_idx, timestamp_s=timestamp_s, state=result.follow_state)
                    if result.follow_state is not None
                    else None
                )

                with self._lock:
                    self.state.frames_processed = frame_idx + 1
                    self.state.last_timestamp_s = float(timestamp_s)
                    self.state.last_status = result.status
                    self.state.last_detections = [detection_to_record(det) for det in result.detections]
                    self.state.last_follow = follow_payload
                    self.state.latest_frame_jpeg = encode_image(result.rendered, ext=".jpg", jpeg_quality=85)

                frame_idx += 1

            summary = finalize_video_run(
                start_t=start_t,
                frame_count=frame_idx,
                pipeline=pipeline,
                follow_estimator=follow_estimator,
            )
            with self._lock:
                self.state.running = False
                self.state.ended_at = time.time()
                self.state.summary = video_run_result_to_record(
                    result=summary,
                    pipeline=pipeline,
                    output_path=self.state.output_path,
                )
        except Exception as exc:
            with self._lock:
                self.state.running = False
                self.state.ended_at = time.time()
                self.state.error = str(exc)
                self.state.summary = {
                    "frames": int(frame_idx),
                    "pipeline": pipeline_stats_to_record(pipeline),
                }
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()


class ConeApiService:
    def __init__(self, app_ctx: AppContext):
        self.app_ctx = app_ctx
        self._streams: Dict[str, ApiStreamWorker] = {}
        self._streams_lock = threading.Lock()

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y", "sim"}

    def _query_value(self, query: Optional[Dict[str, List[str]]], key: str) -> Optional[str]:
        if not query:
            return None
        values = query.get(key)
        if not values:
            return None
        return values[0]

    def _client_cwd(self, body: Optional[Dict[str, Any]], headers: Optional[Dict[str, str]]) -> Optional[str]:
        body = body or {}
        headers = headers or {}
        return (
            body.get("client_cwd")
            or body.get("cwd")
            or headers.get("X-Client-Cwd")
            or headers.get("X-Cwd")
        )

    def _resolve_output_path(
        self,
        requested: Optional[str],
        stem: str,
        suffix: str,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Path:
        if requested:
            return resolve_client_filesystem_path(str(requested), client_cwd=self._client_cwd(body, headers))
        self.app_ctx.default_output_dir.mkdir(parents=True, exist_ok=True)
        return (self.app_ctx.default_output_dir / f"{stem}_{uuid4().hex}{suffix}").resolve()

    def _build_image_payload(
        self,
        frame: np.ndarray,
        source_label: str,
        return_image: bool,
        output_path: Optional[Path] = None,
        return_image_ext: str = ".jpg",
    ) -> Dict[str, object]:
        pipeline = self.app_ctx.create_pipeline()
        follow_estimator = self.app_ctx.create_follow_estimator()
        result = run_frame_job(
            frame=frame,
            frame_index=0,
            timestamp_s=0.0,
            pipeline=pipeline,
            model_tag=self.app_ctx.model_tag,
            overlay_ctx=self.app_ctx.overlay_ctx,
            show_orange_ratio=self.app_ctx.show_orange_ratio,
            follow_estimator=follow_estimator,
        )

        payload: Dict[str, object] = {
            "ok": True,
            "kind": "image",
            "source": source_label,
            "model": self.app_ctx.model_tag,
            "count": len(result.detections),
            "status": result.status,
            "detections": [detection_to_record(det) for det in result.detections],
            "pipeline": pipeline_stats_to_record(pipeline),
            "image_shape": {
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
            },
        }
        if result.follow_state is not None:
            payload["follow"] = follow_state_to_record(frame_index=0, timestamp_s=0.0, state=result.follow_state)
        rendered_output_ext = (
            normalize_image_output_format(output_path.suffix)
            if output_path is not None and output_path.suffix
            else normalize_image_output_format(return_image_ext)
        )
        rendered_output_bytes: Optional[bytes] = None
        if output_path is not None:
            rendered_output_bytes = encode_image(result.rendered, ext=rendered_output_ext, jpeg_quality=90)
            ensure_parent(output_path)
            with output_path.open("wb") as f:
                f.write(rendered_output_bytes)
            payload["output_image"] = str(output_path)
            payload["output_image_format"] = rendered_output_ext.lstrip(".")
        if return_image:
            if rendered_output_bytes is None or rendered_output_ext != normalize_image_output_format(return_image_ext):
                rendered_output_bytes = encode_image(
                    result.rendered,
                    ext=normalize_image_output_format(return_image_ext),
                    jpeg_quality=90,
                )
            payload["rendered_image_base64"] = base64.b64encode(
                rendered_output_bytes
            ).decode("ascii")
            payload["rendered_image_format"] = normalize_image_output_format(return_image_ext).lstrip(".")
        return payload

    def detect_image(
        self,
        *,
        image_bytes: Optional[bytes] = None,
        source: Optional[str] = None,
        body: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, List[str]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        headers = headers or {}
        body = body or {}
        client_cwd = self._client_cwd(body, headers)
        return_image = self._as_bool(
            body.get("return_image", self._query_value(query, "return_image")),
            default=True,
        )
        output_path = resolve_requested_image_output(
            body.get("output", self._query_value(query, "output")),
            client_cwd=client_cwd,
            output_format=body.get("output_format", self._query_value(query, "output_format")),
        )
        return_image_ext = normalize_image_output_format(
            body.get("output_format", self._query_value(query, "output_format"))
        )

        if source is None:
            source = body.get("source")

        if source:
            image_path = Path(resolve_client_source_ref(str(source), client_cwd=client_cwd))
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Falha ao abrir imagem: {image_path}")
            return self._build_image_payload(
                frame=frame,
                source_label=str(image_path),
                return_image=return_image,
                output_path=output_path,
                return_image_ext=return_image_ext,
            )

        if image_bytes is None and body.get("image_b64"):
            image_bytes = base64.b64decode(str(body["image_b64"]))

        if image_bytes is None:
            raise ValueError("Envie bytes da imagem ou um campo 'source' para /detect/image")

        frame = decode_image_bytes(image_bytes)
        source_label = headers.get("X-Filename") or str(body.get("filename") or "upload_image")
        return self._build_image_payload(
            frame=frame,
            source_label=source_label,
            return_image=return_image,
            output_path=output_path,
            return_image_ext=return_image_ext,
        )

    def detect_video(
        self,
        *,
        video_bytes: Optional[bytes] = None,
        source: Optional[str] = None,
        body: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, List[str]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        del query
        headers = headers or {}
        body = body or {}

        if source is None:
            source = body.get("source")

        max_frames = body.get("max_frames")
        max_frames = int(max_frames) if max_frames is not None else None
        output_path = self._resolve_output_path(
            body.get("output"),
            stem="api_video",
            suffix=".mp4",
            body=body,
            headers=headers,
        )

        temp_input_path: Optional[Path] = None
        try:
            if source:
                source_label = resolve_client_source_ref(str(source), client_cwd=self._client_cwd(body, headers))
            else:
                if video_bytes is None and body.get("video_b64"):
                    video_bytes = base64.b64decode(str(body["video_b64"]))
                if video_bytes is None:
                    raise ValueError("Envie bytes do vídeo ou um campo 'source' para /detect/video")
                filename_hint = headers.get("X-Filename") or str(body.get("filename") or "upload.mp4")
                suffix = Path(filename_hint).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(prefix="cone_api_in_", suffix=suffix, delete=False) as tmp:
                    tmp.write(video_bytes)
                    temp_input_path = Path(tmp.name)
                source_label = str(temp_input_path)

            pipeline = self.app_ctx.create_pipeline()
            follow_estimator = self.app_ctx.create_follow_estimator()
            result = process_video_or_camera(
                source=source_label,
                out_path=output_path,
                pipeline=pipeline,
                model_tag=self.app_ctx.model_tag,
                overlay_ctx=self.app_ctx.overlay_ctx,
                show=False,
                show_orange_ratio=self.app_ctx.show_orange_ratio,
                camera_width=self.app_ctx.camera_width,
                camera_height=self.app_ctx.camera_height,
                buffer_size=max(0, self.app_ctx.buffer_size),
                drop_grabs=max(0, self.app_ctx.drop_grabs),
                follow_estimator=follow_estimator,
                follow_log_file=None,
                max_frames=max_frames,
            )
            payload = video_run_result_to_record(result=result, pipeline=pipeline, output_path=output_path)
            payload.update(
                {
                    "ok": True,
                    "kind": "video",
                    "source": source_label,
                    "model": self.app_ctx.model_tag,
                    "output_video": str(output_path),
                }
            )
            return payload
        finally:
            if temp_input_path is not None and temp_input_path.exists():
                temp_input_path.unlink()

    def start_stream(
        self,
        *,
        source: str,
        body: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, List[str]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        del query
        headers = headers or {}
        body = body or {}
        source = str(source).strip()
        if not source:
            raise ValueError("Campo 'source' é obrigatório para iniciar um stream.")

        record = self._as_bool(body.get("record"), default=False) or bool(body.get("output"))
        output_path = (
            self._resolve_output_path(
                body.get("output"),
                stem="api_stream",
                suffix=".mp4",
                body=body,
                headers=headers,
            )
            if record
            else None
        )
        max_frames = body.get("max_frames")
        max_frames = int(max_frames) if max_frames is not None else None
        max_seconds = body.get("max_seconds")
        max_seconds = float(max_seconds) if max_seconds is not None else None
        source = resolve_client_source_ref(source, client_cwd=self._client_cwd(body, headers))

        stream_id = f"stream-{uuid4().hex[:12]}"
        worker = ApiStreamWorker(
            app_ctx=self.app_ctx,
            stream_id=stream_id,
            source=source,
            output_path=output_path,
            max_frames=max_frames,
            max_seconds=max_seconds,
        )
        with self._streams_lock:
            self._streams[stream_id] = worker
        worker.start()
        return worker.snapshot()

    def get_stream_status(self, stream_id: str) -> Dict[str, object]:
        with self._streams_lock:
            worker = self._streams.get(stream_id)
        if worker is None:
            raise KeyError(stream_id)
        return worker.snapshot()

    def get_stream_frame(self, stream_id: str) -> bytes:
        with self._streams_lock:
            worker = self._streams.get(stream_id)
        if worker is None:
            raise KeyError(stream_id)
        return worker.latest_frame()

    def stop_stream(self, stream_id: str) -> Dict[str, object]:
        with self._streams_lock:
            worker = self._streams.get(stream_id)
        if worker is None:
            raise KeyError(stream_id)
        return worker.stop()

    def close(self) -> None:
        with self._streams_lock:
            workers = list(self._streams.values())
        for worker in workers:
            worker.stop(join_timeout=2.0)


def create_api_server(bind_addr: str, service) -> ThreadingHTTPServer:
    host, port = parse_api_bind(bind_addr)

    class ApiHandler(BaseHTTPRequestHandler):
        server_version = "cone-detector-v4-api/1.0"

        def log_message(self, format: str, *args) -> None:
            print(f"[API] {self.address_string()} - {format % args}")

        def _send_json(self, status_code: int, payload: Dict[str, object]) -> None:
            data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_bytes(self, status_code: int, data: bytes, content_type: str) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0") or "0")
            return self.rfile.read(length) if length > 0 else b""

        def _parse_body_and_query(self) -> Tuple[bytes, Dict[str, Any], Dict[str, List[str]]]:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            raw_body = self._read_body()
            body: Dict[str, Any] = {}
            content_type = self.headers.get("Content-Type", "")
            if raw_body and content_type.startswith("application/json"):
                body = json.loads(raw_body.decode("utf-8"))
            return raw_body, body, query

        def _handle_error(self, exc: Exception) -> None:
            if isinstance(exc, KeyError):
                self._send_json(404, {"ok": False, "error": f"Recurso não encontrado: {exc.args[0]}"})
                return
            if isinstance(exc, ValueError):
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            if isinstance(exc, RuntimeError):
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(500, {"ok": False, "error": str(exc)})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            try:
                if parsed.path == "/healthz":
                    self._send_json(200, {"ok": True})
                    return
                if parsed.path == "/config":
                    payload = {"ok": True}
                    if hasattr(service, "app_ctx"):
                        payload.update(
                            {
                                "model": service.app_ctx.model_tag,
                                "profile": service.app_ctx.overlay_ctx.profile,
                                "runtime": {
                                    "conf": service.app_ctx.runtime.conf,
                                    "iou": service.app_ctx.runtime.iou,
                                    "max_det": service.app_ctx.runtime.max_det,
                                    "det_every": service.app_ctx.runtime.det_every,
                                    "prefilter_enabled": service.app_ctx.runtime.prefilter_enabled,
                                    "prefilter_min_ratio": service.app_ctx.runtime.prefilter_min_ratio,
                                    "min_box_orange_ratio": service.app_ctx.runtime.min_box_orange_ratio,
                                },
                            }
                        )
                    self._send_json(200, payload)
                    return
                if len(parts) == 2 and parts[0] == "streams":
                    self._send_json(200, service.get_stream_status(parts[1]))
                    return
                if len(parts) == 3 and parts[0] == "streams" and parts[2] == "frame":
                    self._send_bytes(200, service.get_stream_frame(parts[1]), "image/jpeg")
                    return
                self._send_json(404, {"ok": False, "error": f"Rota não encontrada: {parsed.path}"})
            except Exception as exc:
                self._handle_error(exc)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            try:
                raw_body, body, query = self._parse_body_and_query()
                header_map = {k: v for k, v in self.headers.items()}

                if parsed.path == "/detect/image":
                    source = body.get("source") if body else None
                    image_bytes = None if source else (raw_body or None)
                    payload = service.detect_image(
                        image_bytes=image_bytes,
                        source=source,
                        body=body,
                        query=query,
                        headers=header_map,
                    )
                    self._send_json(200, payload)
                    return

                if parsed.path == "/detect/video":
                    source = body.get("source") if body else None
                    video_bytes = None if source else (raw_body or None)
                    payload = service.detect_video(
                        video_bytes=video_bytes,
                        source=source,
                        body=body,
                        query=query,
                        headers=header_map,
                    )
                    self._send_json(200, payload)
                    return

                if parsed.path == "/streams/start":
                    source = str(body.get("source") or "")
                    payload = service.start_stream(source=source, body=body, query=query, headers=header_map)
                    self._send_json(200, payload)
                    return

                if len(parts) == 3 and parts[0] == "streams" and parts[2] == "stop":
                    self._send_json(200, service.stop_stream(parts[1]))
                    return

                self._send_json(404, {"ok": False, "error": f"Rota não encontrada: {parsed.path}"})
            except Exception as exc:
                self._handle_error(exc)

    class ApiServer(ThreadingHTTPServer):
        daemon_threads = True

        def server_close(self) -> None:
            if hasattr(service, "close"):
                service.close()
            super().server_close()

    return ApiServer((host, port), ApiHandler)


def default_output_for_source(source: Path) -> Path:
    out_dir = SCRIPT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_image_path(source):
        return out_dir / f"{source.stem}_detected{source.suffix}"
    if is_video_path(source):
        return out_dir / f"{source.stem}_detected.mp4"
    if source.is_dir():
        return out_dir / f"{source.name}_detected"
    return out_dir / "camera_detected.mp4"


def build_follow_cfg(args: argparse.Namespace) -> Optional[FollowConfig]:
    follow_enabled = bool(args.follow) or bool(args.follow_jsonl.strip())
    if not follow_enabled:
        return None
    return FollowConfig(
        camera_hfov_deg=args.camera_hfov_deg,
        camera_fx_px=args.camera_fx_px,
        camera_fy_px=args.camera_fy_px,
        cone_height_m=args.cone_height_m,
        target_distance_m=args.target_distance_m,
        target_box_height_ratio=args.target_box_height_ratio,
        ema_alpha=args.error_ema_alpha,
        kp_ang=args.follow_kp_ang,
        kd_ang=args.follow_kd_ang,
        kp_dist=args.follow_kp_dist,
        max_w=args.follow_max_w,
        max_v=args.follow_max_v,
        deadband_heading_deg=args.follow_deadband_deg,
        deadband_dist=args.follow_deadband_dist,
        v_slowdown_heading_deg=args.follow_v_slowdown_deg,
    )


def build_app_context(args: argparse.Namespace, runtime: RuntimePreset, selected_model: ModelSpec) -> AppContext:
    detector = YoloOnnxDetector(
        model=selected_model,
        conf_thres=runtime.conf,
        iou_thres=runtime.iou,
        max_det=runtime.max_det,
        threads=runtime.threads,
        spinning=args.enable_spinning,
        graph_opt=args.graph_opt,
    )

    use_roi = bool(args.roi.strip())
    roi_norm = parse_roi(args.roi) if use_roi else (0.0, 0.0, 1.0, 1.0)
    overlay_ctx = OverlayContext(
        profile=args.profile,
        runtime=runtime,
        use_roi=use_roi,
        roi_norm=roi_norm,
    )

    if args.api.strip():
        if args.output.strip():
            api_out = Path(args.output).expanduser().resolve()
            default_output_dir = api_out if not api_out.suffix else api_out.parent
        else:
            default_output_dir = SCRIPT_DIR / "outputs" / "api"
    else:
        default_output_dir = SCRIPT_DIR / "outputs"

    return AppContext(
        runtime=runtime,
        detector=detector,
        model_tag=selected_model.tag,
        overlay_ctx=overlay_ctx,
        show_orange_ratio=args.show_orange_ratio,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        buffer_size=max(0, args.buffer_size),
        drop_grabs=max(0, args.drop_grabs),
        follow_cfg=build_follow_cfg(args),
        default_output_dir=default_output_dir.resolve(),
    )


def print_runtime_info(args: argparse.Namespace, app_ctx: AppContext) -> None:
    print(f"[INFO] Modelo: {app_ctx.model_tag}")
    print(
        "[INFO] Params: "
        f"profile={args.profile}, conf={app_ctx.runtime.conf}, iou={app_ctx.runtime.iou}, "
        f"max_det={app_ctx.runtime.max_det}, det_every={app_ctx.runtime.det_every}, "
        f"prefilter={app_ctx.runtime.prefilter_enabled}, "
        f"prefilter_min_ratio={app_ctx.runtime.prefilter_min_ratio}, "
        f"min_box_orange_ratio={app_ctx.runtime.min_box_orange_ratio}, "
        f"threads={app_ctx.runtime.threads}, graph_opt={args.graph_opt}"
    )
    if app_ctx.overlay_ctx.use_roi:
        print(f"[INFO] ROI: {app_ctx.overlay_ctx.roi_norm}")
    if app_ctx.follow_cfg is not None:
        print(
            "[INFO] Follow: "
            f"hfov={args.camera_hfov_deg}, fx={args.camera_fx_px}, fy={args.camera_fy_px}, "
            f"cone_h={args.cone_height_m}, target_dist={args.target_distance_m}, "
            f"target_box_ratio={args.target_box_height_ratio}, "
            f"k_ang=({args.follow_kp_ang},{args.follow_kd_ang}), "
            f"k_dist={args.follow_kp_dist}, max_vw=({args.follow_max_v},{args.follow_max_w})"
        )


def run_cli_mode(args: argparse.Namespace, app_ctx: AppContext) -> None:
    pipeline = app_ctx.create_pipeline()
    follow_estimator = app_ctx.create_follow_estimator()
    source = args.source.strip()
    source_path = Path(source)

    out_arg = args.output.strip()
    if out_arg:
        output_path = Path(out_arg).expanduser().resolve()
    else:
        output_path = default_output_for_source(source_path if source_path.exists() else Path("camera"))

    print_runtime_info(args, app_ctx)

    txt_lines: List[str] = []
    follow_log_file = None
    follow_jsonl_path: Optional[Path] = None
    if follow_estimator is not None and args.follow_jsonl.strip():
        follow_jsonl_path = Path(args.follow_jsonl).expanduser().resolve()
        ensure_parent(follow_jsonl_path)
        follow_log_file = follow_jsonl_path.open("w", encoding="utf-8")
        meta = {
            "type": "meta",
            "model": app_ctx.model_tag,
            "profile": args.profile,
            "source": args.source,
            "follow_enabled": True,
        }
        follow_log_file.write(json.dumps(meta, ensure_ascii=True) + "\n")

    try:
        if source_path.exists() and source_path.is_file() and is_image_path(source_path):
            out_file = output_path if not output_path.is_dir() else output_path / source_path.name
            count, log_line = process_image_file(
                image_path=source_path,
                out_path=out_file,
                pipeline=pipeline,
                model_tag=app_ctx.model_tag,
                overlay_ctx=app_ctx.overlay_ctx,
                show=args.show,
                show_orange_ratio=app_ctx.show_orange_ratio,
                follow_estimator=follow_estimator,
                follow_log_file=follow_log_file,
            )
            print(f"[OK] {log_line}")
            print(f"[OK] Resultado salvo em: {out_file}")
            txt_lines.append(log_line)
            txt_lines.append(f"total={count}")

        elif source_path.exists() and source_path.is_dir():
            out_dir = output_path
            if out_dir.suffix:
                raise RuntimeError("Para source em diretório, --output deve ser diretório.")
            logs, total = process_image_dir(
                in_dir=source_path,
                out_dir=out_dir,
                pipeline=pipeline,
                model_tag=app_ctx.model_tag,
                overlay_ctx=app_ctx.overlay_ctx,
                show_orange_ratio=app_ctx.show_orange_ratio,
                follow_estimator=follow_estimator,
                follow_log_file=follow_log_file,
            )
            for line in logs:
                print(f"[OK] {line}")
            print(f"[OK] Total de cones detectados: {total}")
            print(f"[OK] Imagens salvas em: {out_dir}")
            txt_lines.extend(logs)
            txt_lines.append(f"total={total}")

        else:
            out_video = output_path
            if out_video.exists() and out_video.is_dir():
                out_video = out_video / "camera_detected.mp4"
            result = process_video_or_camera(
                source=source,
                out_path=out_video,
                pipeline=pipeline,
                model_tag=app_ctx.model_tag,
                overlay_ctx=app_ctx.overlay_ctx,
                show=args.show,
                show_orange_ratio=app_ctx.show_orange_ratio,
                camera_width=app_ctx.camera_width,
                camera_height=app_ctx.camera_height,
                buffer_size=app_ctx.buffer_size,
                drop_grabs=app_ctx.drop_grabs,
                follow_estimator=follow_estimator,
                follow_log_file=follow_log_file,
            )
            for line in result.logs:
                print(f"[INFO] {line}")
            print(f"[OK] Vídeo salvo em: {out_video}")
            txt_lines.extend(result.logs)
    finally:
        if follow_log_file is not None:
            follow_log_file.close()
            if follow_jsonl_path is not None:
                print(f"[OK] Follow JSONL salvo em: {follow_jsonl_path}")

    if follow_estimator is not None:
        fs = follow_estimator.summary()
        txt_lines.append(f"follow_found_ratio={fs['found_ratio']:.3f}")
        txt_lines.append(f"follow_mean_abs_heading_deg={fs['mean_abs_heading_deg']:.3f}")
        txt_lines.append(f"follow_mean_abs_distance_error={fs['mean_abs_distance_error_ctrl']:.3f}")

    if args.save_txt:
        txt_path = Path(args.save_txt).expanduser().resolve()
        write_txt_log(txt_path, txt_lines)
        print(f"[OK] Log TXT salvo em: {txt_path}")


def run_api_mode(args: argparse.Namespace, app_ctx: AppContext) -> None:
    print_runtime_info(args, app_ctx)
    service = ConeApiService(app_ctx)
    server = create_api_server(args.api.strip(), service)
    host, port = server.server_address[:2]
    print(f"[INFO] API escutando em http://{host}:{port}")
    print("[INFO] Endpoints: GET /healthz, POST /detect/image, POST /detect/video, POST /streams/start")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[INFO] Encerrando API...")
    finally:
        server.server_close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_root = Path(args.model_root).expanduser().resolve()
    catalog = ModelCatalog(model_root)

    if args.list_models:
        print("Modelos encontrados:")
        for spec in catalog.list_models():
            print(f"- {spec.family}/{spec.variant} -> {spec.path} | input={spec.input_w}x{spec.input_h} | output={spec.output_shape}")
        return

    if not args.api.strip() and not args.source.strip():
        parser.error("--source é obrigatório quando --api não é usado.")

    runtime = resolve_runtime(args)
    selected_model = catalog.select(runtime.family, runtime.variant)
    app_ctx = build_app_context(args=args, runtime=runtime, selected_model=selected_model)

    if args.api.strip():
        run_api_mode(args, app_ctx)
        return

    run_cli_mode(args, app_ctx)


if __name__ == "__main__":
    main()
