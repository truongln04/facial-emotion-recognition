# -*- coding: utf-8 -*-
import os
import shutil
import cv2
import numpy as np
import random
from src.utils.face import get_face_detector, detect_faces_bgr


def preprocess_for_dataset(bgr):
    # ====== (1) BGR -> GRAY theo công thức ======
    B = bgr[:, :, 0].astype(np.float32)
    G = bgr[:, :, 1].astype(np.float32)
    R = bgr[:, :, 2].astype(np.float32)

    gray = 0.114 * B + 0.587 * G + 0.299 * R
    gray = gray.astype(np.uint8)

    # ====== (2) CLAHE thủ công ======
    h, w = gray.shape
    tile = 8
    tile_h = h // tile
    tile_w = w // tile
    clip_limit = 40  # giống clipLimit=2.0 của OpenCV nhưng dạng thô

    clahe_img = np.zeros_like(gray)

    for ty in range(tile):
        for tx in range(tile):
            y0 = ty * tile_h
            x0 = tx * tile_w
            tile_gray = gray[y0:y0 + tile_h, x0:x0 + tile_w]

            # Histogram
            hist, _ = np.histogram(tile_gray.flatten(), bins=256, range=(0, 256))

            # Clip histogram
            excess = np.sum(np.maximum(hist - clip_limit, 0))
            hist = np.minimum(hist, clip_limit)

            # Redistribute phần bị cắt
            hist += excess // 256

            # CDF
            cdf = np.cumsum(hist).astype(np.float32)
            cdf = (cdf - cdf.min()) * 255.0 / (cdf.max() - cdf.min())
            cdf = cdf.astype(np.uint8)

            # Ánh xạ pixel tile -> equalized
            clahe_img[y0:y0 + tile_h, x0:x0 + tile_w] = cdf[tile_gray]

    # ====== (3) Resize 224×224 bằng Bilinear Interpolation ======
    out_h, out_w = 224, 224
    resized = np.zeros((out_h, out_w), dtype=np.uint8)

    scale_x = w / out_w
    scale_y = h / out_h

    for y in range(out_h):
        for x in range(out_w):
            src_x = x * scale_x
            src_y = y * scale_y

            x0 = int(np.floor(src_x))
            y0 = int(np.floor(src_y))
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)

            a = src_x - x0
            b = src_y - y0

            I00 = gray[y0, x0]
            I10 = gray[y0, x1]
            I01 = gray[y1, x0]
            I11 = gray[y1, x1]

            value = (I00 * (1 - a) * (1 - b) +
                     I10 * a * (1 - b) +
                     I01 * (1 - a) * b +
                     I11 * a * b)

            resized[y, x] = int(value)

    return resized


def split_and_save(data_dir="data"):
    """
    Tiền xử lý dữ liệu FER2013 đã có sẵn train/test.
    - raw/train: dữ liệu huấn luyện
    - raw/test: dữ liệu kiểm thử
    -> processed/train, processed/val, processed/test
    Lưu ý: val sẽ được tách từ train theo tỷ lệ 15%.
    """
    raw_dir = os.path.join(data_dir, "raw")
    proc_dir = os.path.join(data_dir, "processed")
    cascade = get_face_detector()

    # clean processed
    if os.path.isdir(proc_dir):
        shutil.rmtree(proc_dir)
    os.makedirs(proc_dir, exist_ok=True)

    # xử lý train -> tách train/val
    train_dir = os.path.join(raw_dir, "train")
    if os.path.isdir(train_dir):
        for label in os.listdir(train_dir):
            src_dir = os.path.join(train_dir, label)
            if not os.path.isdir(src_dir):
                continue
            imgs = [os.path.join(src_dir, fn) for fn in os.listdir(src_dir)
                    if fn.lower().endswith((".jpg", ".jpeg", ".png"))]
            random.shuffle(imgs)
            n_total = len(imgs)
            n_train = int(0.85 * n_total)  # 85% train, 15% val

            for i, path in enumerate(imgs):
                bgr = cv2.imread(path)
                if bgr is None:
                    continue
                faces = detect_faces_bgr(bgr, cascade)
                if len(faces) == 0:
                    crop = bgr
                else:
                    x, y, w, h = faces[0]
                    h_img, w_img = bgr.shape[:2]
                    x = max(0, x);
                    y = max(0, y)
                    w = min(w, w_img - x);
                    h = min(h, h_img - y)
                    crop = bgr[y:y + h, x:x + w]
                if crop is None or crop.size == 0:
                    continue

                # tiền xử lý
                proc_img = preprocess_for_dataset(crop)

                split = "train" if i < n_train else "val"
                out_dir = os.path.join(proc_dir, split, label)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, os.path.basename(path))
                cv2.imwrite(out_path, proc_img)

    # xử lý test (giữ nguyên nhãn, thêm tiền xử lý)
    test_dir = os.path.join(raw_dir, "test")
    if os.path.isdir(test_dir):
        for label in os.listdir(test_dir):
            src_dir = os.path.join(test_dir, label)
            if not os.path.isdir(src_dir):
                continue
            out_dir = os.path.join(proc_dir, "test", label)
            os.makedirs(out_dir, exist_ok=True)

            for fn in os.listdir(src_dir):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(src_dir, fn)
                    bgr = cv2.imread(path)
                    if bgr is None:
                        continue
                    faces = detect_faces_bgr(bgr, cascade)
                    if len(faces) == 0:
                        crop = bgr
                    else:
                        x, y, w, h = faces[0]
                        h_img, w_img = bgr.shape[:2]
                        x = max(0, x);
                        y = max(0, y)
                        w = min(w, w_img - x);
                        h = min(h, h_img - y)
                        crop = bgr[y:y + h, x:x + w]
                    if crop is None or crop.size == 0:
                        continue

                    # tiền xử lý
                    proc_img = preprocess_for_dataset(crop)

                    out_path = os.path.join(out_dir, fn)
                    cv2.imwrite(out_path, proc_img)

    print("Preprocess done. Dataset saved to:", proc_dir)

