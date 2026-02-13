#!/usr/bin/env python3
"""
ConeDetector Fusion - Com seleção de "cone 1" por posição
Ajuste: CONE_ORDER = "left_to_right" ou "right_to_left"
"""

import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import onnxruntime as ort
import cv2
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Literal
from collections import deque


# CONFIGURAÇÃO: Direção de ordenação
CONE_ORDER: Literal["left_to_right", "right_to_left"] = "left_to_right"  # Mude aqui


@dataclass
class Detection:
    class_name: str
    confidence: float
    confidence_error: float
    bbox: Dict[str, float]
    bbox_error: Dict[str, float]
    stability: float
    method: str
    position_rank: int = 0  # Rank por posição (1 = primeiro)
    
    def to_dict(self):
        return asdict(self)


class ConeDetectorFusion:
    def __init__(
        self,
        model_path: str = "coneslayer-simplified.onnx",
        num_threads: int = 8,
        conf_thres: float = 0.54,
        img_size: int = 416,
        cone_order: Literal["left_to_right", "right_to_left"] = "left_to_right"
    ):
        self.conf_thres = conf_thres
        self.img_size = img_size
        self.cone_order = cone_order  # Direção de ordenação
        
        # ONNX setup
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path, opts, providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_buffer = np.empty((1, 3, img_size, img_size), dtype=np.float32)
        
        # OpenCV HSV
        self.hsv_lower = np.array([5, 100, 100])
        self.hsv_upper = np.array([25, 255, 255])
        
        self._warmup()
        print(f"[Init] Ordenação: {self.cone_order}")
    
    def _warmup(self):
        dummy = np.random.randn(1, 3, self.img_size, self.img_size).astype(np.float32)
        for _ in range(3):
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
    
    def detect_onnx(self, img: np.ndarray) -> List[Dict]:
        input_tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        preds = outputs[0][0]
        mask = preds[:, 4] > self.conf_thres
        filtered = preds[mask]
        
        if len(filtered) == 0:
            return []
        
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
                    'x': float(x * scale_x),  # Centro X em pixels
                    'y': float(y * scale_y),
                    'w': float(w * scale_x),
                    'h': float(h * scale_y)
                }
            })
        
        return detections
    
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
                'bbox': {
                    'x': float(x + w/2),  # Centro X
                    'y': float(y + h/2),
                    'w': float(w),
                    'h': float(h)
                }
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
    
    def rank_by_position(self, detections: List[Detection]) -> List[Detection]:
        """
        Ordena detectões por posição X e atribui position_rank
        """
        if not detections:
            return []
        
        # Ordenar por coordenada X (centro do bbox)
        if self.cone_order == "left_to_right":
            # Menor X = mais à esquerda = rank 1
            sorted_dets = sorted(detections, key=lambda d: d.bbox['x'])
        else:
            # Maior X = mais à direita = rank 1
            sorted_dets = sorted(detections, key=lambda d: d.bbox['x'], reverse=True)
        
        # Atribuir ranks
        for i, det in enumerate(sorted_dets, 1):
            det.position_rank = i
        
        return sorted_dets
    
    def fuse(self, onnx_dets: List[Dict], opencv_dets: List[Dict], 
             img_shape: Tuple[int, int]) -> List[Detection]:
        
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
        
        # Ordenar por posição e atribuir ranks
        return self.rank_by_position(fused)
    
    def detect(self, img: np.ndarray) -> Tuple[List[Detection], float, Dict]:
        t0 = time.perf_counter()
        
        onnx_dets = self.detect_onnx(img)
        opencv_dets = self.detect_opencv(img)
        fused = self.fuse(onnx_dets, opencv_dets, img.shape[:2])
        
        total_ms = (time.perf_counter() - t0) * 1000
        
        stats = {
            'total_ms': round(total_ms, 2),
            'total_cones': len(fused),
            'cone_1_method': fused[0].method if fused else None,
            'cone_1_x': fused[0].bbox['x'] if fused else None,
            'order': self.cone_order
        }
        
        return fused, total_ms, stats
    
    def to_json(self, detections: List[Detection], stats: Dict, detailed: bool = True) -> str:
        """Exporta JSON com destaque para cone 1"""
        
        # Separar cone 1 dos outros
        cone_1 = None
        other_cones = []
        for d in detections:
            if d.position_rank == 1:
                cone_1 = d.to_dict()
            else:
                other_cones.append(d.to_dict())
        
        result = {
            'timestamp': time.time(),
            'cone_1': cone_1,  # Destaque especial
            'other_cones': other_cones,
            'all_cones': [d.to_dict() for d in detections],  # Lista completa ordenada
            'count': len(detections),
            'config': {
                'order': self.cone_order,
                'detailed': detailed
            },
            'performance': stats
        }
        
        return json.dumps(result, indent=2)


def main():
    import sys
    
    # Instanciar com direção configurada
    detector = ConeDetectorFusion(
        num_threads=8,
        cone_order=CONE_ORDER  # "left_to_right" ou "right_to_left"
    )
    
    img_path = sys.argv[1] if len(sys.argv) > 1 else "ima.jpg"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Erro: {img_path} não encontrado")
        return
    
    print(f"\nProcessando: {img.shape}")
    print(f"Modo: {CONE_ORDER}")
    
    dets, ms, stats = detector.detect(img)
    
    # Print JSON
    print(detector.to_json(dets, stats))
    
    # Print resumo
    if dets:
        cone_1 = next((d for d in dets if d.position_rank == 1), None)
        if cone_1:
            print(f"\n=== CONE 1 ===")
            print(f"Posição: ({cone_1.bbox['x']:.1f}, {cone_1.bbox['y']:.1f})")
            print(f"Confiança: {cone_1.confidence:.2f} ± {cone_1.confidence_error:.2f}")
            print(f"Método: {cone_1.method}")
            print(f"Dimensões: {cone_1.bbox['w']:.1f} x {cone_1.bbox['h']:.1f}")
    
    # Visualização
    for det in dets:
        x, y, w, h = det.bbox['x'], det.bbox['y'], det.bbox['w'], det.bbox['h']
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Destaque especial para cone 1
        if det.position_rank == 1:
            color = (0, 255, 255)  # Ciano para cone 1
            thickness = 3
            label = f"CONE 1 ({det.method}) {det.confidence:.2f}"
        else:
            color = (0, 0, 255) if det.method == 'onnx' else (0, 255, 0)
            thickness = 2
            label = f"#{det.position_rank} {det.method} {det.confidence:.2f}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite("/tmp/resultado_cone1.jpg", img)
    print(f"\nImagem: /tmp/resultado_cone1.jpg")


if __name__ == "__main__":
    main()
