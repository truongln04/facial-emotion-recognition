# 😀 Facial Emotion Recognition

Hệ thống nhận diện cảm xúc khuôn mặt sử dụng Deep Learning với PyTorch, hỗ trợ huấn luyện mô hình CNN/ResNet18 trên tập dữ liệu FER2013 và dự đoán cảm xúc từ ảnh hoặc webcam thông qua giao diện PyQt6.

## 📌 Giới thiệu

Facial Emotion Recognition là dự án ứng dụng Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision) nhằm nhận diện cảm xúc của con người thông qua biểu cảm khuôn mặt.

Hệ thống sử dụng mô hình học sâu được xây dựng bằng PyTorch để phân loại các trạng thái cảm xúc phổ biến như:

* Angry 😠
* Disgust 🤢
* Fear 😨
* Happy 😄
* Sad 😢
* Surprise 😲
* Neutral 😐

Dự án hỗ trợ toàn bộ quy trình từ tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá kết quả đến triển khai ứng dụng giao diện người dùng bằng PyQt6.

---

# 🎯 Mục tiêu

* Xây dựng hệ thống nhận diện cảm xúc khuôn mặt theo thời gian thực.
* Huấn luyện mô hình CNN/ResNet18 trên tập dữ liệu FER2013.
* Phân loại chính xác các trạng thái cảm xúc cơ bản.
* Cung cấp giao diện trực quan cho người dùng.
* Hỗ trợ nghiên cứu và học tập trong lĩnh vực Deep Learning và Computer Vision.

---

# ✨ Chức năng chính

## 📂 Quản lý dữ liệu

* Tiền xử lý dữ liệu FER2013.
* Chia tập Train/Test.
* Chuẩn hóa dữ liệu đầu vào.
* Crop khuôn mặt tự động.
* Tăng cường dữ liệu (Data Augmentation).

## 🤖 Huấn luyện mô hình

* Huấn luyện CNN tùy chỉnh.
* Huấn luyện ResNet18 Transfer Learning.
* Lưu trọng số mô hình.
* Theo dõi Loss và Accuracy.

## 📊 Đánh giá mô hình

* Accuracy Score.
* Classification Report.
* Confusion Matrix.
* Đánh giá trên tập Test.

## 🖥️ Giao diện người dùng

* Nhận diện cảm xúc từ ảnh.
* Nhận diện cảm xúc từ webcam.
* Hiển thị khuôn mặt được phát hiện.
* Hiển thị xác suất dự đoán.
* Xuất kết quả dự đoán.

---

# 🛠️ Công nghệ sử dụng

## Ngôn ngữ lập trình

* Python 3.10+

## Deep Learning

* PyTorch
* Torchvision

## Computer Vision

* OpenCV
* Haar Cascade / MTCNN

## Xử lý dữ liệu

* NumPy
* Pandas
* Scikit-learn

## Giao diện

* PyQt6

## Visualization

* Matplotlib
* Seaborn

---

# 📁 Cấu trúc thư mục

```text
facial-emotion-recognition/
├── app/
│   └── gui.py

├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── test/
│   └── processed/

├── models/
│   └── emotion_resnet18.pt

├── scripts/
│   ├── preprocess.py
│   ├── train_cnn.py
│   └── evaluate.py

├── src/
│   ├── utils/
│   │   ├── preprocess.py
│   │   ├── dataset.py
│   │   └── face.py
│   │
│   └── models/
│       └── cnn.py

├── main.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

Dự án sử dụng tập dữ liệu FER2013.

Thêm dữ liệu vào thư mục:

```text
data/
└── raw/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    │
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── sad/
        ├── surprise/
        └── neutral/
```

Sau khi thêm dataset, chạy bước tiền xử lý để tạo dữ liệu trong thư mục `processed`.

---

# ⚙️ Cài đặt

## 1. Clone Repository

```bash
git clone https://github.com/truongln04/facial-emotion-recognition.git
cd facial-emotion-recognition
```

## 2. Cài đặt thư viện

Nâng cấp pip:

```bash
python -m pip install --upgrade pip
```

Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Sử dụng

## Bước 1: Tiền xử lý dữ liệu

```bash
python main.py --mode preprocess
```

Chương trình sẽ:

* Đọc dữ liệu từ `data/raw`
* Tiền xử lý ảnh
* Chuẩn hóa dữ liệu
* Lưu kết quả vào `data/processed`

---

## Bước 2: Huấn luyện mô hình

Huấn luyện 30 epochs:

```bash
python main.py --mode train_cnn --epochs 30 --lr 1e-4 --batch 64
```

Huấn luyện 50 epochs:

```bash
python main.py --mode train_cnn --epochs 50 --lr 1e-4 --batch 64
```

Huấn luyện 100 epochs:

```bash
python main.py --mode train_cnn --epochs 100 --lr 1e-4 --batch 64
```

Trọng số mô hình sau khi huấn luyện sẽ được lưu trong thư mục:

```text
models/
```

---

## Bước 3: Đánh giá mô hình

```bash
python main.py --mode eval
```

Kết quả:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Bước 4: Chạy giao diện GUI

```bash
python main.py --mode gui
```

Các chức năng:

* Mở ảnh từ máy tính.
* Nhận diện cảm xúc từ webcam.
* Hiển thị kết quả dự đoán.
* Xuất kết quả nhận diện.

---

# 📈 Kết quả

Mô hình được đánh giá trên tập dữ liệu FER2013 với các chỉ số:

* Accuracy
* Precision
* Recall
* F1-Score

Kết quả chi tiết có thể được xem thông qua báo cáo đánh giá và ma trận nhầm lẫn được sinh ra bởi chương trình.

---

# 🔮 Hướng phát triển

* Triển khai mô hình trên Web.
* Tích hợp YOLO Face Detection.
* Hỗ trợ video thời gian thực.
* Tối ưu tốc độ suy luận.
* Triển khai trên thiết bị Edge hoặc Mobile.
* Hỗ trợ nhiều bộ dữ liệu cảm xúc hơn.

---

# 👨‍💻 Tác giả

**Lưu Nguyên Trường**

Sinh viên Công nghệ Thông tin

GitHub: https://github.com/truongln04

---

# 📄 License

Dự án được phát triển cho mục đích học tập, nghiên cứu và thực hành các kỹ thuật Deep Learning trong nhận diện cảm xúc khuôn mặt.
