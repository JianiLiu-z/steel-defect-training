import os
import cv2
import random
import shutil
import numpy as np

# 小类ID（按你的类别顺序修改）
TARGET_CLASSES = [6, 7, 8]

IMAGE_DIR = "datasets/GC10-DET/images/train"
LABEL_DIR = "datasets/GC10-DET/labels/train"

OUT_IMAGE_DIR = IMAGE_DIR
OUT_LABEL_DIR = LABEL_DIR

AUG_PER_IMAGE = 3


def adjust_brightness(img):
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


def rotate_image(img, angle):
    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angle,
        1.0
    )

    return cv2.warpAffine(img, M, (w, h))


def has_target_class(label_path):

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        cls_id = int(line.split()[0])

        if cls_id in TARGET_CLASSES:
            return True

    return False


image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".png", ".jpeg"))
]

count = 0

for img_name in image_files:

    print("Processing:", img_name)

    img_path = os.path.join(IMAGE_DIR, img_name)

    label_name = os.path.splitext(img_name)[0] + ".txt"

    label_path = os.path.join(LABEL_DIR, label_name)

    if not os.path.exists(label_path):
        continue

    if not has_target_class(label_path):
        continue

    img = cv2.imread(img_path)

    for i in range(AUG_PER_IMAGE):

        aug = img.copy()

        # flip
        if random.random() < 0.5:
            aug = cv2.flip(aug, 1)

        # brightness
        aug = adjust_brightness(aug)

        # noise
        aug = add_noise(aug)

        # rotate
        angle = random.uniform(-5, 5)

        aug = rotate_image(aug, angle)

        new_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"

        out_img_path = os.path.join(
            OUT_IMAGE_DIR,
            new_name
        )

        out_label_path = os.path.join(
            OUT_LABEL_DIR,
            new_name.replace(".jpg", ".txt")
        )

        cv2.imwrite(out_img_path, aug)

        shutil.copy(
            label_path,
            out_label_path
        )

        count += 1

print(f"Generated {count} augmented images.")