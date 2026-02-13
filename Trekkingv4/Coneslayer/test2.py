#!/usr/bin/env python3
"""
ConeDetector Fusion - Otimizado para 7-8 FPS (~140ms)
Correção: Forçar paralelismo ONNX
"""

import os
# FORÇAR variáveis ANTES de importar onnxruntime
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

import onnxruntime as ort
import cv2
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from collections import deque


@dataclass
class Detection:
    class_name: str
    confidence: float
    confidence_error: float
    bbox: Dict[str, float]
    bbox_error: Dict[str, float]
    stability: float
    method: str
    
    def to_dict(self):
        return asdict(self)


class ConeDetectorFusion:
    def __init__(
        self,
        model_path: str = "coneslayer-simplified.onnx",
        num_threads: int = 8,
        conf_thres: float = 0.54,
        img_size: int = 416
    ):
        self.conf_thres = conf_thres
        self.img_size = img_size
        
        # Configuração AGRESSIVA de paralelismo
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads  # Threads por operação
        opts.inter_op_num_threads = num_threads   # Threads entre operações
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # CRÍTICO!
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = True
        
        # Log para debug
        print(f"[Init] ONNX Runtime version: {ort.__version__}")
        print(f"[Init] Available providers: {ort.get_available_providers()}")
        print(f"[Init] Intra threads: {opts.intra_op_num_threads}")
        print(f"[Init] Inter threads: {opts.inter_op_num_threads}")
        
        self.session = ort.InferenceSession(
            model_path, 
            opts, 
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_buffer = np.empty((1, 3, img_size, img_size), dtype=np.float32)
        
        # OpenCV HSV
        self.hsv_lower = np.array([5, 100, 100])
        self.hsv_upper = np.array([25, 255, 255])
        self.detection_history = deque(maxlen=5)
        
        self._warmup()
        print(f"[Init] Pronto - threads confirmadas")
    
    def _warmup(self):
        dummy = np.random.randn(1, 3, self.img_size, self.img_size).astype(np.float32)
        for _ in range(5):
            self.session.run(None, {self.input_name: dummy})
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_LINEAR)
        
        img_rgb = img[:, :, ::-1].transpose(2, 0, 1)
        np.multiply(
            np.ascontiguousarray(img_rgb, dtype=np.float32),
            1.0/255.0,
            out=self.input_buffer[0]
        )
        return self.input_buffer
    
    def detect_onnx(self, img: np.ndarray) -> Tuple[List[Dict], float]:
        """Retorna detecções + tempo"""
        t0 = time.perf_counter()
        
        input_tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        t1 = time.perf_counter()
        onnx_ms = (t1 - t0) * 1000
        
        preds = outputs[0][0]
        mask = preds[:, 4] > self.conf_thres
        filtered = preds[mask]
        
        if len(filtered) == 0:
            return [], onnx_ms
        
        boxes = filtered[:, :4]
        scores = filtered[:, 4] * filtered[:, 5:].max(axis=1)
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_thres,
            0.5
        )
        
        detections = []
        scale_x = img.shape[1] / self.img_size
        scale_y = img.shape[0] / self.img_size
        
        for idx in indices.flatten() if len(indices) > 0 else []:
            x, y, w, h = boxes[idx]
            detections.append({
                'confidence': float(scores[idx]),
                'bbox': {
                    'x': float(x * scale_x),
                    'y': float(y * scale_y),
                    'w': float(w * scale_x),
                    'h': float(h * scale_y)
                }
            })
        
        return detections, onnx_ms
    
    def detect_opencv(self, img: np.ndarray) -> List[Dict]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / w if w > 0 else 0
            if aspect < 0.8 or aspect > 3.0:
                continue
            
            area_score = min(area / 5000, 1.0)
            aspect_score = 1.0 - abs(aspect - 1.5) / 1.5
            confidence = 0.3 + 0.4 * area_score + 0.3 * max(0, aspect_score)
            
            detections.append({
                'confidence': float(confidence),
                'bbox': {'x': float(x + w/2), 'y': float(y + h/2), 
                        'w': float(w), 'h': float(h)},
                'aspect': float(aspect)
            })
        
        return detections
    
    def calculate_iou(self, b1: Dict, b2: Dict) -> float:
        x1 = max(b1['x'] - b1['w']/2, b2['x'] - b2['w']/2)
        y1 = max(b1['y'] - b1['h']/2, b2['y'] - b2['h']/2)
        x2 = min(b1['x'] + b1['w']/2, b2['x'] + b2['w']/2)
        y2 = min(b1['y'] + b1['h']/2, b2['y'] + b2['h']/2)
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = b1['w'] * b1['h']
        area2 = b2['w'] * b2['h']
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def fuse(self, onnx_dets: List[Dict], opencv_dets: List[Dict], 
             img_shape: Tuple[int, int], onnx_ms: float) -> Tuple[List[Detection], Dict]:
        
        t0 = time.perf_counter()
        
        fused = []
        used_cv = set()
        
        # Fusão ONNX + OpenCV
        for onnx_det in onnx_dets:
            best_match = None
            best_iou = 0
            
            for i, cv_det in enumerate(opencv_dets):
                if i in used_cv:
                    continue
                iou = self.calculate_iou(onnx_det['bbox'], cv_det['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = i
            
            if best_match is not None:
                cv_det = opencv_dets[best_match]
                used_cv.add(best_match)
                
                w_onnx = onnx_det['confidence']
                w_cv = cv_det['confidence'] * 0.8
                
                total_w = w_onnx + w_cv
                fused_bbox = {
                    'x': (onnx_det['bbox']['x'] * w_onnx + cv_det['bbox']['x'] * w_cv) / total_w,
                    'y': (onnx_det['bbox']['y'] * w_onnx + cv_det['bbox']['y'] * w_cv) / total_w,
                    'w': (onnx_det['bbox']['w'] * w_onnx + cv_det['bbox']['w'] * w_cv) / total_w,
                    'h': (onnx_det['bbox']['h'] * w_onnx + cv_det['bbox']['h'] * w_cv) / total_w,
                }
                
                bbox_error = {
                    'x': abs(onnx_det['bbox']['x'] - cv_det['bbox']['x']) / 2,
                    'y': abs(onnx_det['bbox']['y'] - cv_det['bbox']['y']) / 2,
                    'w': abs(onnx_det['bbox']['w'] - cv_det['bbox']['w']) / 2,
                    'h': abs(onnx_det['bbox']['h'] - cv_det['bbox']['h']) / 2,
                }
                
                fused_conf = (w_onnx + w_cv) / 1.8
                conf_error = abs(onnx_det['confidence'] - cv_det['confidence']) / 2
                
                fused.append(Detection(
                    class_name='cone',
                    confidence=round(fused_conf, 4),
                    confidence_error=round(conf_error, 4),
                    bbox={k: round(v, 2) for k, v in fused_bbox.items()},
                    bbox_error={k: round(v, 2) for k, v in bbox_error.items()},
                    stability=round(best_iou, 3),
                    method='fusion'
                ))
            else:
                fused.append(Detection(
                    class_name='cone',
                    confidence=round(onnx_det['confidence'], 4),
                    confidence_error=0.15,
                    bbox={k: round(v, 2) for k, v in onnx_det['bbox'].items()},
                    bbox_error={'x': 5.0, 'y': 5.0, 'w': 8.0, 'h': 8.0},
                    stability=0.0,
                    method='onnx'
                ))
        
        # Adicionar OpenCV não usados
        for i, cv_det in enumerate(opencv_dets):
            if i not in used_cv and cv_det['confidence'] > 0.6:
                fused.append(Detection(
                    class_name='cone',
                    confidence=round(cv_det['confidence'] * 0.7, 4),
                    confidence_error=0.25,
                    bbox={k: round(v, 2) for k, v in cv_det['bbox'].items()},
                    bbox_error={'x': 10.0, 'y': 10.0, 'w': 15.0, 'h': 15.0},
                    stability=0.0,
                    method='opencv'
                ))
        
        fusion_ms = (time.perf_counter() - t0) * 1000
        
        stats = {
            'onnx_ms': round(onnx_ms, 2),
            'fusion_ms': round(fusion_ms, 2),
            'onnx_count': len(onnx_dets),
            'opencv_count': len(opencv_dets),
            'fusion_count': sum(1 for d in fused if d.method == 'fusion')
        }
        
        return fused, stats
    
    def detect(self, img: np.ndarray) -> Tuple[List[Detection], float, Dict]:
        """Pipeline completo"""
        t0 = time.perf_counter()
        
        # ONNX (domina o tempo)
        onnx_dets, onnx_ms = self.detect_onnx(img)
        
        # OpenCV (rápido, paralelizável)
        opencv_dets = self.detect_opencv(img)
        
        # Fusão
        fused, stats = self.fuse(onnx_dets, opencv_dets, img.shape[:2], onnx_ms)
        
        total_ms = (time.perf_counter() - t0) * 1000
        stats['total_ms'] = round(total_ms, 2)
        
        return fused, total_ms, stats
    
    def to_json(self, detections: List[Detection], stats: Dict) -> str:
        result = {
            'timestamp': time.time(),
            'detections': [d.to_dict() for d in detections],
            'count': len(detections),
            'performance': stats
        }
        return json.dumps(result, indent=2)


def main():
    import sys
    
    detector = ConeDetectorFusion(num_threads=8)
    
    img_path = sys.argv[1] if len(sys.argv) > 1 else "ima.jpg"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Erro: {img_path} não encontrado")
        return
    
    print(f"\nProcessando: {img.shape}")
    
    # Benchmark de 10 runs para média real
    times = []
    for i in range(10):
        dets, ms, stats = detector.detect(img)
        times.append(ms)
        if i == 0:
            print(detector.to_json(dets, stats))
    
    print(f"\n=== BENCHMARK 10 runs ===")
    print(f"Média: {np.mean(times):.1f}ms ({1000/np.mean(times):.1f} FPS)")
    print(f"Min: {np.min(times):.1f}ms, Max: {np.max(times):.1f}ms")


if __name__ == "__main__":
    main()
