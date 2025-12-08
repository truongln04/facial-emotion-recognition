import cv2
from facenet_pytorch import MTCNN

def get_face_detector():
    # Khởi tạo MTCNN, keep_all=True để phát hiện nhiều khuôn mặt
    detector = MTCNN(keep_all=True)
    return detector

def bgr_to_rgb_manual(bgr):
    """
    CHUYỂN BGR → RGB THEO CÔNG THỨC HOÁN ĐỔI KÊNH
    RGB = [R, G, B] = [bgr[...,2], bgr[...,1], bgr[...,0]]
    KHÔNG dùng cvtColor.
    """
    # Lấy từng kênh theo vị trí
    B = bgr[:, :, 0]
    G = bgr[:, :, 1]
    R = bgr[:, :, 2]

    # Ghép lại theo thứ tự mới (R, G, B)
    rgb = cv2.merge([R, G, B])
    return rgb

def detect_faces_bgr(bgr, detector):
    # MTCNN yêu cầu ảnh RGB, tự chuyển theo công thức (KHÔNG dùng hàm có sẵn)
    rgb = bgr_to_rgb_manual(bgr)

    # MTCNN detect
    boxes, _ = detector.detect(rgb)

    faces = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            faces.append((x1, y1, x2 - x1, y2 - y1))  # (x,y,w,h)

    return faces
