import cv2
import numpy as np
import onnxruntime as ort

# ============= Config =============
MODEL_PATH = "coneslayer-simplified.onnx"
IMAGE_PATH = "0_image1.png"
OUTPUT_PATH = "output.png"

CONF_THRESHOLD = 0.54   # abaixe p/ 0.3 se quiser debug
IOU_THRESHOLD = 0.5
INPUT_W, INPUT_H = 416, 416
# ==================================

def preprocess(img):
    img_resized = cv2.resize(img, (INPUT_W, INPUT_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0)

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
    area_b = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def nms(boxes, scores, thresh):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        idxs = idxs[1:]
        if idxs.size == 0:
            break
        idxs = np.array([j for j in idxs if iou(boxes[i], boxes[j]) < thresh])
    return keep

def detect_and_normalize(det, img_w, img_h):
    """
    Recebe um vetor det[0:6] (os primeiros 6 valores) e retorna uma caixa xyxy em pixels:
    [x1, y1, x2, y2] (inteiros, relativos à imagem original img_w x img_h).
    Detecta formatos:
      - cx,cy,w,h,conf,class (normalizado ou em pixels)
      - x1,y1,x2,y2,conf,class (normalizado ou em pixels)
    Estratégia heurística:
      - se coords > 1.5 e <= INPUT_W/INPUT_H: tratamos como coordenadas em escala 416 (pixels do input)
      - se coords > INPUT_W: tratamos como coordenadas em pixels da imagem original
      - se todos <=1: se terceiro valor > primeiro valor --> provavelmente xyxy normalizado; senão -> xywh normalizado
    """
    x0, x1, x2, x3 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
    # Heurísticas:
    max_coord = max(x0, x1, x2, x3)

    # Caso: coordenadas já em pixels relativas à imagem original (valores > INPUT_W)
    if max_coord > max(img_w, img_h):
        # assumimos já xyxy em pixels
        x1p = int(round(x0)); y1p = int(round(x1)); x2p = int(round(x2)); y2p = int(round(x3))
        return [x1p, y1p, x2p, y2p]

    # Caso: coordenadas em escala do input (0..416) — treat as pixels on 416 grid
    if max_coord > 1.5 and max_coord <= max(INPUT_W, INPUT_H):
        # decidir se é xyxy ou cx,cy,w,h: para xyxy geralmente x2>x1.
        if x2 > x0 and x3 > x1:
            # xyxy em escala INPUT
            x1p = int(round(x0 * (img_w / INPUT_W)))
            y1p = int(round(x1 * (img_h / INPUT_H)))
            x2p = int(round(x2 * (img_w / INPUT_W)))
            y2p = int(round(x3 * (img_h / INPUT_H)))
            return [x1p, y1p, x2p, y2p]
        else:
            # então assumimos cx,cy,w,h em escala INPUT
            cx = x0; cy = x1; w = x2; h = x3
            x1p = int(round((cx - w/2) * (img_w / INPUT_W)))
            y1p = int(round((cy - h/2) * (img_h / INPUT_H)))
            x2p = int(round((cx + w/2) * (img_w / INPUT_W)))
            y2p = int(round((cy + h/2) * (img_h / INPUT_H)))
            return [x1p, y1p, x2p, y2p]

    # Caso: coordenadas normalizadas [0..1]
    # distinguir xyxy vs cx,yw: se x2 > x0 assume xyxy normalizado
    if max_coord <= 1.0:
        if x2 > x0 and x3 > x1:
            # xyxy normalizado
            x1p = int(round(x0 * img_w))
            y1p = int(round(x1 * img_h))
            x2p = int(round(x2 * img_w))
            y2p = int(round(x3 * img_h))
            return [x1p, y1p, x2p, y2p]
        else:
            # xywh normalizado
            cx = x0; cy = x1; w = x2; h = x3
            x1p = int(round((cx - w/2) * img_w))
            y1p = int(round((cy - h/2) * img_h))
            x2p = int(round((cx + w/2) * img_w))
            y2p = int(round((cy + h/2) * img_h))
            return [x1p, y1p, x2p, y2p]

    # Fallback defensivo: normalize by INPUT size then map to image
    cx, cy, w, h = x0, x1, x2, x3
    x1p = int(round((cx - w/2) * (img_w / INPUT_W)))
    y1p = int(round((cy - h/2) * (img_h / INPUT_H)))
    x2p = int(round((cx + w/2) * (img_w / INPUT_W)))
    y2p = int(round((cy + h/2) * (img_h / INPUT_H)))
    return [x1p, y1p, x2p, y2p]

# ------------- Main -------------
img0 = cv2.imread(IMAGE_PATH)
if img0 is None:
    raise RuntimeError("Imagem não encontrada em " + IMAGE_PATH)
orig_h, orig_w = img0.shape[:2]

inp = preprocess(img0)

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {input_name: inp})

# Use only aggregated output [1, 10647, 6] (caso exista)
pred = outputs[0]
if pred.ndim == 3 and pred.shape[0] == 1:
    pred = pred[0]  # [N, 6]

print("DEBUG: pred shape:", pred.shape)

# Estatísticas rápidas (ajuda muito a entender o formato)
if pred.size > 0:
    arr = np.array(pred)
    coords = arr[:, :4]
    confs = arr[:, 4]
    print("DEBUG: confs min/max:", float(confs.min()), float(confs.max()))
    print("DEBUG: coords min/max:", float(coords.min()), float(coords.max()))
    # print os primeiros 6 detections > 0.1 (raw)
    sample = arr[arr[:,4] > 0.1][:6]
    print("DEBUG: sample detections (first up to 6):")
    for s in sample:
        print(" ", np.array2string(s, precision=3, separator=", "))
else:
    print("DEBUG: saída vazia do modelo")
    pred = np.zeros((0,6), dtype=np.float32)

boxes = []
scores = []

for det in pred:
    # det pode ser [cx,cy,w,h,conf,cls] ou [x1,y1,x2,y2,conf,cls]
    conf = float(det[4])
    if conf < CONF_THRESHOLD:
        continue

    # converte para xyxy em pixels relativos à imagem original
    box = detect_and_normalize(det, orig_w, orig_h)
    # clamp
    x1, y1, x2, y2 = box
    x1 = max(0, min(orig_w-1, x1))
    y1 = max(0, min(orig_h-1, y1))
    x2 = max(0, min(orig_w-1, x2))
    y2 = max(0, min(orig_h-1, y2))
    if x2 <= x1 or y2 <= y1:
        continue

    boxes.append([x1, y1, x2, y2])
    scores.append(conf)

boxes = np.array(boxes) if boxes else np.zeros((0,4), dtype=int)
scores = np.array(scores) if len(scores) else np.zeros((0,))

print(f"DEBUG: after threshold -> {len(scores)} candidates")

keep = nms(boxes, scores, IOU_THRESHOLD)
print(f"[OK] {len(keep)} cones após NMS (threshold={CONF_THRESHOLD}, iou={IOU_THRESHOLD})")

# desenha
out = img0.copy()
for i in keep:
    x1, y1, x2, y2 = boxes[i].astype(int)
    conf = scores[i]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"cone {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0,255,0), -1)
    cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

cv2.imwrite(OUTPUT_PATH, out)
print(f"[OK] Resultado salvo em {OUTPUT_PATH}")
