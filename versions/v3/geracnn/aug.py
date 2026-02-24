import cv2
import os
import glob
import albumentations as A
from tqdm import tqdm
import random

# pastas
images_dir = "dataset_raw/train/images"
labels_dir = "dataset_raw/train/labels"
output_dir = "dataset_aug"

os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

# define augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# pega todas as imagens
images = glob.glob(os.path.join(images_dir, "*.jpg"))

for img_path in tqdm(images):
    img_name = os.path.basename(img_path)
    label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))

    # lê label
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                bboxes.append([x, y, w, h])
                class_labels.append(class_id)

    # lê imagem
    image = cv2.imread(img_path)

    for i in range(3):  # cria 3 imagens augmentadas por original
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        out_img_path = os.path.join(output_dir, "images", f"{img_name[:-4]}_aug{i}.jpg")
        out_label_path = os.path.join(output_dir, "labels", f"{img_name[:-4]}_aug{i}.txt")

        cv2.imwrite(out_img_path, aug_image)
        with open(out_label_path, "w") as f:
            for cls, bbox in zip(aug_labels, aug_bboxes):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")
