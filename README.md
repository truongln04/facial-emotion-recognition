## Cấu trúc thư mục

facial-emotion-recognition/
├── app/
│   └── gui.py               # GUI PyQt6 (hiển thị webcam/ảnh, dự đoán, export)
│
├── data/
│   ├── raw/                 # dữ liệu gốc (FER2013 hoặc ảnh riêng)
│   └── processed/           # dữ liệu đã tiền xử lý (tự sinh bởi scripts/preprocess.py)
│
├── models/
│   └── emotion_resnet18.pt  # trọng số mô hình đã train (ResNet18 hoặc CNN)
│
├── scripts/
│   ├── preprocess.py        # tách train/val/test, crop mặt, tiền xử lý dataset
│   ├── train_cnn.py         # train mô hình CNN/ResNet18
│   └── evaluate.py          # đánh giá mô hình (confusion matrix, classification report)
│
├── src/
│   ├── utils/
│   │   ├── preprocess.py    # tiền xử lý cho ảnh tĩnh & webcam (grayscale, CLAHE, resize, normalize)
│   │   ├── dataset.py       # PyTorch Dataset/DataLoader, augmentation, label mapping
│   │   └── face.py          # phát hiện mặt (Haar Cascade hoặc MTCNN)
│   │
│   └── models/
│       └── cnn.py           # định nghĩa mô hình CNN/ResNet18 bằng PyTorch
│
├── main.py                  # entry point để chạy app hoặc CLI
├── README.md                # mô tả dự án, cách chạy
├── requirements.txt         # danh sách thư viện cần cài

## Run

1) Cài đặt
   python -m pip install --upgrade pip
pip install -r requirements.txt


2) Tiền xử lý dữ liệu (tùy dataset bạn có)
   python main.py --mode preprocess

3) Train CNN
   python main.py --mode train_cnn --epochs 30 --lr 1e-4 --batch 64
   python main.py --mode train_cnn --epochs 50 --lr 1e-4 --batch 64
   python main.py --mode train_cnn --epochs 100 --lr 1e-4 --batch 64

4) Đánh giá
   python main.py --mode eval

5) GUI
   python main.py --mode gui
