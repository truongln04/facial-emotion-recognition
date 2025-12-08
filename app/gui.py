# -*- coding: utf-8 -*-
import sys, os, time, csv
import cv2, numpy as np
import torch
import torch.nn.functional as F
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QProgressBar, QListWidget, QListWidgetItem
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from src.utils.preprocess import preprocess_image_resnet

from src.utils.face import get_face_detector, detect_faces_bgr
from src.models.resnet import EmotionResNet
from src.utils.dataset import EMOTIONS

EXPORT_DIR = os.path.join("D:/FER_Results")
os.makedirs(EXPORT_DIR, exist_ok=True)

class CameraThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(float)

    def __init__(self, cam_index=0):
        super().__init__()
        self.cam_index = cam_index
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        prev = time.time()
        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            dt = now - prev
            prev = now
            fps = 1.0/dt if dt>0 else 0.0
            self.fps_signal.emit(fps)
            self.frame_signal.emit(frame)
            self.msleep(30)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class FER_GUI(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.setWindowTitle("FER - PyQt6 + Face Box + Export")
        self.setGeometry(150, 80, 1100, 600)

        # Load face detector + model
        self.cascade = get_face_detector()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionResNet(num_classes=7).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: Display + Controls
        left = QVBoxLayout()
        self.display_label = QLabel()
        self.display_label.setFixedSize(640,480)
        self.display_label.setStyleSheet("background-color:black;")
        left.addWidget(self.display_label, alignment=Qt.AlignmentFlag.AlignCenter)

        ctrl_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Webcam")
        self.btn_stop = QPushButton("Stop")
        self.btn_load = QPushButton("Load Image")
        self.btn_export = QPushButton("Export CSV/Images")
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_load)
        ctrl_layout.addWidget(self.btn_export)
        left.addLayout(ctrl_layout)

        main_layout.addLayout(left)

        # Right: Confidence bars + info + history
        right = QVBoxLayout()
        self.top_label = QLabel("No prediction")
        self.top_label.setStyleSheet("font-size:18px;font-weight:bold;")
        right.addWidget(self.top_label)

        self.bars = {}
        for emo in EMOTIONS:
            row = QHBoxLayout()
            lbl = QLabel(emo)
            lbl.setFixedWidth(80)
            bar = QProgressBar()
            bar.setRange(0,100)
            bar.setValue(0)
            bar.setFormat("%p%")
            row.addWidget(lbl)
            row.addWidget(bar)
            right.addLayout(row)
            self.bars[emo] = bar

        info_row = QHBoxLayout()
        self.fps_label = QLabel("FPS:0.0")
        self.infer_label = QLabel("Infer:0 ms")
        info_row.addWidget(self.fps_label)
        info_row.addWidget(self.infer_label)
        right.addLayout(info_row)

        right.addWidget(QLabel("History (timestamp - label - %)"))
        self.history = QListWidget()
        right.addWidget(self.history)
        main_layout.addLayout(right)

        # Thread & signals
        self.cam_thread = CameraThread()
        self.cam_thread.frame_signal.connect(self.on_frame)
        self.cam_thread.fps_signal.connect(lambda f: self.fps_label.setText(f"FPS:{f:.1f}"))
        self.btn_start.clicked.connect(self.start_cam)
        self.btn_stop.clicked.connect(self.stop_cam)
        self.btn_load.clicked.connect(self.load_image)
        self.btn_export.clicked.connect(self.export_all)

        self.history_data = []
        self.frame_count = 0
        self.last_label = None   # lưu cảm xúc trước đó

    def start_cam(self):
        if not self.cam_thread.isRunning():
            self.cam_thread.start()

    def stop_cam(self):
        if self.cam_thread.isRunning():
            self.cam_thread.stop()

    def load_image(self):
        # Hộp thoại chọn ảnh
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return

        # Đọc ảnh gốc
        frame = cv2.imread(path)

        # Phát hiện khuôn mặt
        faces = detect_faces_bgr(frame, self.cascade)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_bgr = frame[y:y + h, x:x + w]

            # Tiền xử lý + dự đoán
            probs = self.infer_probs(face_bgr, source="dataset")

            # Vẽ kết quả lên ảnh
            self.draw_result(frame, (x, y, w, h), probs)

        # Hiển thị ảnh lên GUI
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        qt_img = qt_img.scaled(self.display_label.width(), self.display_label.height(),
                               Qt.AspectRatioMode.KeepAspectRatio)
        self.display_label.setPixmap(QPixmap.fromImage(qt_img))

    def infer_probs(self, face_bgr, source="webcam"):
        start = time.time()
        # dùng preprocess cho ResNet
        x = preprocess_image_resnet(face_bgr).unsqueeze(0).to(self.device)  # (1,3,224,224)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        infer_ms = int((time.time() - start) * 1000)
        self.infer_label.setText(f"Infer:{infer_ms} ms")
        return probs

    def draw_result(self, frame, face_rect, probs):
        x, y, w, h = face_rect
        idx = int(np.argmax(probs))
        label = EMOTIONS[idx]
        pct = probs[idx] * 100

        # Luôn vẽ box + update bar
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}-{pct:.1f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for i, emo in enumerate(EMOTIONS):
            self.bars[emo].setValue(int(probs[i] * 100))

        # Kiểm tra thay đổi cảm xúc hoặc confidence dao động mạnh
        update = False
        if label != self.last_label:
            update = True
        elif self.last_label == label:
            last_pct = self.history_data[0]["confidence"] if self.history_data else "0.0"
            try:
                delta = abs(float(last_pct) - pct)
                if delta >= 5.0:  # ngưỡng dao động
                    update = True
            except:
                update = True

        if update:
            self.top_label.setText(f"{label} — {pct:.1f}%")
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            hist_line = f"{ts} - {label} - {pct:.1f}%"
            item = QListWidgetItem(hist_line)
            self.history.insertItem(0, item)
            self.history_data.insert(0, {"timestamp": ts, "label": label, "confidence": f"{pct:.1f}"})
            self.last_label = label

    # def process_and_display(self, frame, source="webcam"):
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     # faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    #     faces = detect_faces_bgr(frame, self.cascade)  # nếu self.cascade là MTCNN
    #
    #     if len(faces) > 0:
    #         x,y,w,h = faces[0]
    #         face_bgr = frame[y:y+h, x:x+w]
    #         probs = self.infer_probs(face_bgr, source=source)
    #         self.draw_result(frame, (x,y,w,h), probs)
    #
    #     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     h,w,ch = rgb.shape
    #     bytes_per_line = ch*w
    #     qt_img = QImage(rgb.data,w,h,bytes_per_line,QImage.Format.Format_RGB888)
    #     qt_img = qt_img.scaled(self.display_label.width(), self.display_label.height(),
    #                            Qt.AspectRatioMode.KeepAspectRatio)
    #     self.display_label.setPixmap(QPixmap.fromImage(qt_img))
    def process_and_display(self, frame, source="webcam"):
        faces = detect_faces_bgr(frame, self.cascade)
        for (x, y, w, h) in faces:
            face_bgr = frame[y:y + h, x:x + w]
            probs = self.infer_probs(face_bgr, source=source)
            self.draw_result(frame, (x, y, w, h), probs)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        qt_img = qt_img.scaled(self.display_label.width(), self.display_label.height(),
                               Qt.AspectRatioMode.KeepAspectRatio)
        self.display_label.setPixmap(QPixmap.fromImage(qt_img))

    def on_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % 5 == 0:
            self.process_and_display(frame, source="webcam")
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w,ch = rgb.shape
            bytes_per_line = ch*w
            qt_img = QImage(rgb.data,w,h,bytes_per_line,QImage.Format.Format_RGB888)
            qt_img = qt_img.scaled(self.display_label.width(), self.display_label.height(),
                                   Qt.AspectRatioMode.KeepAspectRatio)

    def export_all(self):
            # Export CSV lịch sử
        csv_path = os.path.join(EXPORT_DIR, "FER_history.csv")
        keys = ["timestamp", "label", "confidence"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.history_data[::-1]:
                writer.writerow(row)

            # Export last frame
        img_path = os.path.join(EXPORT_DIR, "last_frame.jpg")
        pixmap = self.display_label.pixmap()
        if pixmap:
            pixmap.save(img_path)
        self.statusBar().showMessage(f"Exported CSV + last frame to {EXPORT_DIR}", 5000)

def main(model_path):
    app = QApplication(sys.argv)
    win = FER_GUI(model_path=model_path)
    win.show()
    sys.exit(app.exec())
