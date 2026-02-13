#!/usr/bin/env python3
"""
ConeDetector Fusion - Precisão aumentada via fusão ONNX + OpenCV
Retorna JSON com detecções, confianças e erros estimados
Performance: ~140ms (7 FPS) mantido
"""

import onnxruntime as ort
import cv2
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from collections import deque


@dataclass
class Detection:
    """Detecção com metadados de precisão"""
    class_name: str
    confidence: float
    confidence_error: float  # Erro estimado da confiança
    bbox: Dict[str, float]   # x, y, w, h em pixels
    bbox_error: Dict[str, float]  # Erros estimados de cada dimensão
    stability: float         # 0-1, quão estável é a detecção
    method: str              # 'onnx', 'opencv', 'fusion'
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ConeDetectorFusion:
    """
    Fusão ONNX + OpenCV para precisão aumentada
    Sem uso de GPU - otimizado para 7 FPS (~140ms)
    """
    
    def __init__(
        self,
        model_path: str = "coneslayer-simplified.onnx",
        num_threads: int = 8,
        conf_thres: float = 0.54,
        iou_thres: float = 0.5,
        img_size: int = 416,
        history_size: int = 5  # Para estabilidade temporal
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.history_size = history_size
        
        # ONNX Runtime otimizado
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, opts, ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_buffer = np.empty((1, 3, img_size, img_size), dtype=np.float32)
        
        # OpenCV: Detector de cor HSV para cones laranja
        self.hsv_lower = np.array([5, 100, 100])   # Laranja escuro
        self.hsv_upper = np.array([25, 255, 255])  # Laranja claro
        
        # Histórico para estabilidade temporal
        self.detection_history: deque = deque(maxlen=history_size)
        
        # Warmup
        self._warmup()
        
        print(f"[Fusion] ONNX: {model_path}, Threads: {num_threads}")
        print(f"[Fusion] OpenCV HSV: {self.hsv_lower} - {self.hsv_upper}")
        print(f"[Fusion] Pronto")
    
    def _warmup(self):
        """Warmup dos dois métodos"""
        dummy = np.random.randn(1, 3, self.img_size, self.img_size).astype(np.float32)
        for _ in range(3):
            self.session.run(None, {self.input_name: dummy})
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocessamento ONNX"""
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_LINEAR)
        
        img_rgb = img[:, :, ::-1].transpose(2, 0, 1)
        np.multiply(
            np.ascontiguousarray(img_rgb, dtype=np.float32),
            1.0/255.0,
            out=self.input_buffer[0]
        )
        return self.input_buffer
    
    def detect_onnx(self, img: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Detecção via ONNX (neural network)"""
        input_tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        preds = outputs[0][0]
        mask = preds[:, 4] > self.conf_thres
        filtered = preds[mask]
        
        if len(filtered) == 0:
            return [], filtered
        
        boxes = filtered[:, :4]
        scores = filtered[:, 4] * filtered[:, 5:].max(axis=1)
        classes = filtered[:, 5:].argmax(axis=1)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_thres,
            self.iou_thres
        )
        
        detections = []
        scale_x = img.shape[1] / self.img_size
        scale_y = img.shape[0] / self.img_size
        
        for idx in indices.flatten() if len(indices) > 0 else []:
            x, y, w, h = boxes[idx]
            detections.append({
                'class': 'cone',
                'confidence': float(scores[idx]),
                'bbox': {
                    'x': float(x * scale_x),
                    'y': float(y * scale_y),
                    'w': float(w * scale_x),
                    'h': float(h * scale_y)
                },
                'raw_bbox': [float(x), float(y), float(w), float(h)],
                'method': 'onnx'
            })
        
        return detections, filtered
    
    def detect_opencv(self, img: np.ndarray) -> List[Dict]:
        """Detecção via OpenCV (cor + forma)"""
        # Converter para HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Máscara de cor laranja
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Operações morfológicas para limpar ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Filtrar pequenos
                continue
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Proporção típica de cone (altura > largura)
            aspect = h / w if w > 0 else 0
            if aspect < 0.8 or aspect > 3.0:
                continue
            
            # Confiança baseada em área e proporção
            area_score = min(area / 5000, 1.0)
            aspect_score = 1.0 - abs(aspect - 1.5) / 1.5  # Ótimo em 1.5:1
            confidence = 0.3 + 0.4 * area_score + 0.3 * max(0, aspect_score)
            
            detections.append({
                'class': 'cone',
                'confidence': float(confidence),
                'bbox': {'x': float(x + w/2), 'y': float(y + h/2), 
                        'w': float(w), 'h': float(h)},
                'area': float(area),
                'aspect': float(aspect),
                'method': 'opencv'
            })
        
        return detections
    
    def calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calcula IoU entre duas boxes"""
        x1 = max(box1['x'] - box1['w']/2, box2['x'] - box2['w']/2)
        y1 = max(box1['y'] - box1['h']/2, box2['y'] - box2['h']/2)
        x2 = min(box1['x'] + box1['w']/2, box2['x'] + box2['w']/2)
        y2 = min(box1['y'] + box1['h']/2, box2['y'] + box2['h']/2)
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1['w'] * box1['h']
        area2 = box2['w'] * box2['h']
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def fuse_detections(
        self,
        onnx_dets: List[Dict],
        opencv_dets: List[Dict],
        img_shape: Tuple[int, int]
    ) -> List[Detection]:
        """
        Fusão inteligente: combina ONNX (alta precisão) + OpenCV (robustez)
        Estratégia:
        - ONNX tem prioridade (mais preciso)
        - OpenCV complementa quando ONNX falha
        - Fusão quando ambos detectam (média ponderada)
        """
        fused = []
        used_opencv = set()
        
        # 1. Processar detecções ONNX
        for onnx_det in onnx_dets:
            best_match = None
            best_iou = 0
            
            # Procurar match em OpenCV
            for i, cv_det in enumerate(opencv_dets):
                if i in used_opencv:
                    continue
                iou = self.calculate_iou(onnx_det['bbox'], cv_det['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = i
            
            if best_match is not None:
                # FUSÃO: média ponderada pela confiança
                cv_det = opencv_dets[best_match]
                used_opencv.add(best_match)
                
                w_onnx = onnx_det['confidence']
                w_cv = cv_det['confidence'] * 0.8  # Penalidade OpenCV
                
                total_w = w_onnx + w_cv
                
                fused_bbox = {
                    'x': (onnx_det['bbox']['x'] * w_onnx + cv_det['bbox']['x'] * w_cv) / total_w,
                    'y': (onnx_det['bbox']['y'] * w_onnx + cv_det['bbox']['y'] * w_cv) / total_w,
                    'w': (onnx_det['bbox']['w'] * w_onnx + cv_det['bbox']['w'] * w_cv) / total_w,
                    'h': (onnx_det['bbox']['h'] * w_onnx + cv_det['bbox']['h'] * w_cv) / total_w,
                }
                
                # Erros estimados: discrepância entre métodos
                bbox_error = {
                    'x': abs(onnx_det['bbox']['x'] - cv_det['bbox']['x']) / 2,
                    'y': abs(onnx_det['bbox']['y'] - cv_det['bbox']['y']) / 2,
                    'w': abs(onnx_det['bbox']['w'] - cv_det['bbox']['w']) / 2,
                    'h': abs(onnx_det['bbox']['h'] - cv_det['bbox']['h']) / 2,
                }
                
                fused_conf = (w_onnx + w_cv) / (1 + 0.8)  # Normalizado
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
                # Apenas ONNX
                fused.append(Detection(
                    class_name='cone',
                    confidence=round(onnx_det['confidence'], 4),
                    confidence_error=round(0.15, 4),  # Erro estimado padrão ONNX
                    bbox={k: round(v, 2) for k, v in onnx_det['bbox'].items()},
                    bbox_error={'x': 5.0, 'y': 5.0, 'w': 8.0, 'h': 8.0},  # pixels
                    stability=0.0,
                    method='onnx'
                ))
        
        # 2. Adicionar detecções OpenCV não usadas (baixa confiança)
        for i, cv_det in enumerate(opencv_dets):
            if i not in used_opencv and cv_det['confidence'] > 0.6:
                fused.append(Detection(
                    class_name='cone',
                    confidence=round(cv_det['confidence'] * 0.7, 4),  # Penalidade
                    confidence_error=round(0.25, 4),
                    bbox={k: round(v, 2) for k, v in cv_det['bbox'].items()},
                    bbox_error={'x': 10.0, 'y': 10.0, 'w': 15.0, 'h': 15.0},
                    stability=0.0,
                    method='opencv'
                ))
        
        return fused
    
    def calculate_temporal_stability(self, detections: List[Detection]) -> List[Detection]:
        """Calcula estabilidade temporal usando histórico"""
        if not self.detection_history:
            for det in detections:
                det.stability = 0.0
            return detections
        
        # Matching simples por proximidade
        for det in detections:
            stabilities = []
            for past_dets in self.detection_history:
                for past_det in past_dets:
                    dist = np.sqrt(
                        (det.bbox['x'] - past_det.bbox['x'])**2 +
                        (det.bbox['y'] - past_det.bbox['y'])**2
                    )
                    if dist < 20:  # pixels
                        stabilities.append(past_det.stability)
            
            if stabilities:
                det.stability = round(0.7 * det.stability + 0.3 * np.mean(stabilities), 3)
        
        # Atualizar histórico
        self.detection_history.append(detections)
        
        return detections
    
    def detect(self, img: np.ndarray) -> Tuple[List[Detection], float, Dict]:
        """
        Pipeline completo de detecção com fusão
        Retorna: (detecções, tempo_ms, metadados)
        """
        t0 = time.perf_counter()
        
        # 1. Detecção ONNX (neural)
        onnx_dets, raw_output = self.detect_onnx(img)
        t1 = time.perf_counter()
        
        # 2. Detecção OpenCV (cor/forma) - paralelizável
        opencv_dets = self.detect_opencv(img)
        t2 = time.perf_counter()
        
        # 3. Fusão
        fused = self.fuse_detections(onnx_dets, opencv_dets, img.shape[:2])
        t3 = time.perf_counter()
        
        # 4. Estabilidade temporal
        fused = self.calculate_temporal_stability(fused)
        t4 = time.perf_counter()
        
        # Tempos
        total_ms = (t4 - t0) * 1000
        breakdown = {
            'onnx_ms': round((t1 - t0) * 1000, 2),
            'opencv_ms': round((t2 - t1) * 1000, 2),
            'fusion_ms': round((t3 - t2) * 1000, 2),
            'temporal_ms': round((t4 - t3) * 1000, 2),
            'total_ms': round(total_ms, 2)
        }
        
        return fused, total_ms, breakdown
    
    def to_json(self, detections: List[Detection], metadata: Dict) -> str:
        """Exporta resultado completo para JSON"""
        result = {
            'timestamp': time.time(),
            'detections': [d.to_dict() for d in detections],
            'count': len(detections),
            'performance': metadata,
            'fusion_stats': {
                'onnx_count': sum(1 for d in detections if d.method == 'onnx'),
                'opencv_count': sum(1 for d in detections if d.method == 'opencv'),
                'fusion_count': sum(1 for d in detections if d.method == 'fusion')
            }
        }
        return json.dumps(result, indent=2)


def main():
    """Exemplo de uso"""
    import sys
    
    detector = ConeDetectorFusion(num_threads=8)
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "ima.jpg"
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro: Não conseguiu carregar {img_path}")
        return
    
    print(f"\nProcessando: {img.shape}")
    
    # Detecção
    dets, total_ms, meta = detector.detect(img)
    
    # Resultado JSON
    json_output = detector.to_json(dets, meta)
    print("\n=== RESULTADO JSON ===")
    print(json_output)
    
    # Estatísticas
    print(f"\n=== ESTATÍSTICAS ===")
    print(f"Total: {total_ms:.1f}ms")
    print(f"  ONNX: {meta['onnx_ms']:.1f}ms")
    print(f"  OpenCV: {meta['opencv_ms']:.1f}ms")
    print(f"  Fusão: {meta['fusion_ms']:.1f}ms")
    print(f"  Temporal: {meta['temporal_ms']:.1f}ms")
    
    # Visualização (opcional)
    for det in dets:
        x, y, w, h = det.bbox['x'], det.bbox['y'], det.bbox['w'], det.bbox['h']
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Cor por método
        color = {
            'onnx': (255, 0, 0),      # Azul
            'opencv': (0, 255, 0),    # Verde
            'fusion': (0, 0, 255)     # Vermelho
        }.get(det.method, (128, 128, 128))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.method} {det.confidence:.2f} ±{det.confidence_error:.2f}"
        cv2.putText(img, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite("/tmp/resultado_fusion.jpg", img)
    print(f"\nImagem salva: /tmp/resultado_fusion.jpg")


if __name__ == "__main__":
    main()
