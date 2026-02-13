#!/usr/bin/env python3
"""
ConeDetector - Versão final de produção
Otimizado para: ~140ms inference @ 416x416 no Exynos ARM64
"""

import onnxruntime as ort
import cv2
import numpy as np
import json
import struct
from typing import List, Dict, Any, Tuple, Optional


class ConeDetector:
    """Detector otimizado para cones laranja"""
    
    def __init__(
        self,
        model_path: str = "coneslayer-simplified.onnx",
        num_threads: int = 8,
        conf_thres: float = 0.54,
        img_size: int = 416
    ):
        self.conf_thres = conf_thres
        self.img_size = img_size
        
        # ONNX Runtime otimizado
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, opts, ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_buffer = np.empty((1, 3, img_size, img_size), dtype=np.float32)
        
        # Warmup
        dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        for _ in range(3):
            self.session.run(None, {self.input_name: dummy})
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocessamento rápido"""
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_LINEAR)
        
        # BGR->RGB, HWC->CHW, normalize
        img_rgb = img[:, :, ::-1].transpose(2, 0, 1)
        np.multiply(np.ascontiguousarray(img_rgb, dtype=np.float32), 
                   1.0/255.0, out=self.input_buffer[0])
        return self.input_buffer
    
    def postprocess(self, outputs: List[np.ndarray]) -> List[Dict]:
        """Extrair detecções"""
        preds = outputs[0][0]
        mask = preds[:, 4] > self.conf_thres
        filtered = preds[mask]
        
        if len(filtered) == 0:
            return []
        
        boxes = filtered[:, :4]
        scores = filtered[:, 4] * filtered[:, 5:].max(axis=1)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            self.conf_thres, 0.5
        )
        
        return [{
            'class': 'cone',
            'confidence': float(scores[i]),
            'bbox': {
                'x': float(boxes[i][0]),
                'y': float(boxes[i][1]),
                'w': float(boxes[i][2]),
                'h': float(boxes[i][3])
            }
        } for i in indices.flatten()] if len(indices) > 0 else []
    
    def detect(self, img: np.ndarray) -> Tuple[List[Dict], float]:
        """Inference completa"""
        import time
        t0 = time.perf_counter()
        
        input_tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self.postprocess(outputs)
        
        elapsed = (time.perf_counter() - t0) * 1000
        return detections, elapsed


# Exemplo de uso com protocolo binário simples
def process_frame(detector: ConeDetector, img_bytes: bytes) -> bytes:
    """
    Processa frame JPEG e retorna JSON com detecções
    Protocolo: [img_bytes] -> [json_bytes]
    """
    # Decodificar JPEG
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return json.dumps({'error': 'decode_failed'}).encode()
    
    # Detectar
    dets, ms = detector.detect(img)
    
    # Retornar resultado
    result = {
        'detections': dets,
        'time_ms': round(ms, 1),
        'count': len(dets)
    }
    return json.dumps(result).encode()


# Main
if __name__ == "__main__":
    import sys
    
    detector = ConeDetector(num_threads=8)
    
    # Teste simples
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            dets, ms = detector.detect(img)
            print(f"{len(dets)} cones em {ms:.1f}ms")
            for d in dets:
                print(f"  cone: {d['confidence']:.2f}")
        else:
            print("Imagem não encontrada")
    else:
        print("Uso: python3 detector.py <imagem.jpg>")
