import cv2
import numpy as np
import onnxruntime as ort
import os

# ============= Config =============
MODEL_PATH = "best.onnx"
IMAGE_PATH = "dataset"          # <-- pode ser imagem OU pasta
OUTPUT_DIR = "outputs"         # usado quando IMAGE_PATH é pasta
OUTPUT_SINGLE = "output.png"   # usado quando IMAGE_PATH é arquivo

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
INPUT_W, INPUT_H = 320, 320
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
    x0, y0, x1, y1 = map(float, det[:4])
    max_coord = max(x0, y0, x1, y1)

    # pixels absolutos
    if max_coord > max(img_w, img_h):
        return [int(x0), int(y0), int(x1), int(y1)]

    # escala 416
    if max_coord > 1.5:
        if x1 > x0:
            return [
                int(x0 * img_w / INPUT_W),
                int(y0 * img_h / INPUT_H),
                int(x1 * img_w / INPUT_W),
                int(y1 * img_h / INPUT_H),
            ]
        cx, cy, w, h = x0, y0, x1, y1
        return [
            int((cx - w/2) * img_w / INPUT_W),
            int((cy - h/2) * img_h / INPUT_H),
            int((cx + w/2) * img_w / INPUT_W),
            int((cy + h/2) * img_h / INPUT_H),
        ]

    # normalizado [0,1]
    if x1 > x0:
        return [
            int(x0 * img_w),
            int(y0 * img_h),
            int(x1 * img_w),
            int(y1 * img_h),
        ]

    cx, cy, w, h = x0, y0, x1, y1
    return [
        int((cx - w/2) * img_w),
        int((cy - h/2) * img_h),
        int((cx + w/2) * img_w),
        int((cy + h/2) * img_h),
    ]


def process_image(img, sess, input_name):
    orig_h, orig_w = img.shape[:2]
    inp = preprocess(img)
    pred = sess.run(None, {input_name: inp})[0]

    if pred.ndim == 3:
        pred = pred[0]

    boxes, scores = [], []

    for det in pred:
        conf = float(det[4])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = detect_and_normalize(det, orig_w, orig_h)

        x1 = max(0, min(orig_w-1, x1))
        y1 = max(0, min(orig_h-1, y1))
        x2 = max(0, min(orig_w-1, x2))
        y2 = max(0, min(orig_h-1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        scores.append(conf)

    keep = nms(boxes, scores, IOU_THRESHOLD)

    out = img.copy()
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"cone {conf:.2f}"
        cv2.putText(
            out, label, (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    return out, len(keep)


# ============= Main =============
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

if os.path.isdir(IMAGE_PATH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = sorted([
        f for f in os.listdir(IMAGE_PATH)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"[INFO] Processando {len(images)} imagens")

    for name in images:
        path = os.path.join(IMAGE_PATH, name)
        img = cv2.imread(path)
        if img is None:
            continue

        out, n = process_image(img, sess, input_name)
        out_path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(out_path, out)
        print(f"[OK] {name}: {n} cones")

else:
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("Imagem não encontrada")

    out, n = process_image(img, sess, input_name)
    cv2.imwrite(OUTPUT_SINGLE, out)
    print(f"[OK] {n} cones detectados → {OUTPUT_SINGLE}")
