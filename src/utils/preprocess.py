# -*- coding: utf-8 -*-
import numpy as np
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def to_grayscale(bgr):
    """
    Bước 1: Chuyển ảnh BGR sang grayscale bằng công thức tính thủ công.
    Gray = 0.299*R + 0.587*G + 0.114*B
    """
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)

    gray = r * 0.299 + g * 0.587 + b * 0.114
    return gray.clip(0, 255).astype(np.uint8)


def hist_equalization(gray):
    """
    Bước 2: Histogram Equalization viết tay:
    - Tạo histogram
    - Tạo CDF
    - Chuẩn hóa CDF về [0,255]
    - Ánh xạ lại giá trị pixel
    """
    # Histogram 256 mức xám
    hist = np.bincount(gray.ravel(), minlength=256)

    # CDF
    cdf = hist.cumsum()

    # Nếu ảnh toàn đen hoặc không có pixel > 0 → bỏ equalization
    nonzero = cdf[cdf > 0]
    if len(nonzero) == 0:
        return gray

    cdf_nonzero = nonzero[0]

    # Chuẩn hóa CDF về 0–255
    total = gray.size
    if total == cdf_nonzero:   # tránh chia 0
        return gray

    cdf_norm = (cdf - cdf_nonzero) / (total - cdf_nonzero) * 255
    cdf_norm = np.clip(cdf_norm, 0, 255)

    # Map lại pixel
    eq = cdf_norm[gray]
    return eq.astype(np.uint8)


def replicate_rgb(gray):
    """
    Bước 3: Nhân kênh grayscale thành ảnh 3 kênh RGB.
    """
    return np.dstack([gray, gray, gray])


def resize_bilinear(img, new_h=224, new_w=224):
    """
    Bước 4: Resize thủ công bằng nội suy tuyến tính 2 chiều (bilinear interpolation).
    """
    h, w, c = img.shape

    # Tạo grid tọa độ theo pixel-center mapping
    x_scaled = (np.arange(new_w) + 0.5) * (w / new_w) - 0.5
    y_scaled = (np.arange(new_h) + 0.5) * (h / new_h) - 0.5

    x_scaled = np.clip(x_scaled, 0, w - 1)
    y_scaled = np.clip(y_scaled, 0, h - 1)

    x0 = np.floor(x_scaled).astype(np.int32)
    y0 = np.floor(y_scaled).astype(np.int32)

    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    ax = x_scaled - x0
    ay = y_scaled - y0

    out = np.zeros((new_h, new_w, c), dtype=np.float32)

    for ch in range(c):
        I00 = img[y0[:, None], x0[None, :], ch]
        I01 = img[y0[:, None], x1[None, :], ch]
        I10 = img[y1[:, None], x0[None, :], ch]
        I11 = img[y1[:, None], x1[None, :], ch]

        w00 = (1 - ax) * (1 - ay)[:, None]
        w01 = ax * (1 - ay)[:, None]
        w10 = (1 - ax) * ay[:, None]
        w11 = ax * ay[:, None]

        out[..., ch] = (
            w00 * I00 +
            w01 * I01 +
            w10 * I10 +
            w11 * I11
        )

    return out.clip(0, 255).astype(np.uint8)


def to_tensor(img):
    """
    Bước 5: Chuyển ảnh numpy -> tensor torch.
    Scale về [0,1] và đổi sang (C,H,W).
    """
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def normalize_imagenet(tensor):
    """
    Bước 6: Chuẩn hóa theo mean/std ImageNet.
    """
    mean = torch.tensor(IMAGENET_MEAN).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).reshape(3, 1, 1)
    return (tensor - mean) / std


def ensure_dtype(tensor):
    """
    Bước 7: Đưa tensor về float32 để tương thích ResNet.
    """
    return tensor.float()


def preprocess_image_resnet(bgr):
    """
    Pipeline 7 bước GIỮ NGUYÊN:
    1. Grayscale
    2. Histogram Equalization
    3. Replicate RGB
    4. Resize 224×224 (bilinear)
    5. To Tensor
    6. Normalize ImageNet
    7. float32
    """
    g = to_grayscale(bgr)
    eq = hist_equalization(g)
    rgb = replicate_rgb(eq)
    resized = resize_bilinear(rgb, 224, 224)
    tensor = to_tensor(resized)
    tensor = normalize_imagenet(tensor)
    tensor = ensure_dtype(tensor)
    return tensor
