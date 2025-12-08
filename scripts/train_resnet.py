# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.models.resnet import EmotionResNet
from src.utils.dataset import make_loader


# ============================================================
#  HÀM TRAIN 1 EPOCH
# ============================================================
def train_epoch(model, loader, device, criterion, optimizer):
    """
    Thực hiện 1 epoch huấn luyện:
    - Đưa model sang trạng thái train() để bật dropout/batchnorm.
    - Duyệt lần lượt từng batch trong loader.
    - Tính loss, thực hiện backprop và cập nhật tham số.
    - Tính accuracy cho toàn bộ epoch.
    """
    model.train()  # bật chế độ huấn luyện
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        # đưa batch lên GPU/CPU tuỳ thiết bị
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()            # xoá gradient cũ
        logits = model(x)                # forward: dự đoán
        loss = criterion(logits, y)      # tính cross entropy loss
        loss.backward()                  # lan truyền ngược
        optimizer.step()                 # cập nhật trọng số

        # cộng dồn loss và accuracy
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


# ============================================================
#  HÀM EVALUATION
# ============================================================
def eval_epoch(model, loader, device, criterion):
    """
    Evaluation:
    - model.eval() để tắt dropout/batchnorm cập nhật.
    - Không dùng backprop nên đặt trong no_grad().
    - Chỉ forward và tính accuracy + loss.
    """
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, total_correct / total


# ============================================================
#  HÀM CHÍNH TRAINING
# ============================================================
def main(data_dir="data", model_path="models/emotion_resnet.pt",
         epochs=50, lr=1e-4, batch_size=64, patience=10):
    """
    Pipeline huấn luyện đầy đủ:
    - Load dữ liệu từ thư mục data.
    - Khởi tạo model ResNet cho FER.
    - Tạo optimizer, loss, scheduler.
    - Train + evaluate theo epoch.
    - Lưu lại model tốt nhất.
    - Dừng sớm nếu val không cải thiện (Early stopping).
    """
    # chọn GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model ResNet18 tuỳ chỉnh cho nhận dạng cảm xúc
    model = EmotionResNet(num_classes=7).to(device)

    criterion = nn.CrossEntropyLoss()                        # hàm mất mát chính
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # giảm learning rate nếu val_loss đứng yên
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Tạo DataLoader (đã bao gồm resize + normalization cho ResNet)
    train_loader = make_loader(data_dir, "train", batch_size, use_resnet=True, shuffle=True)
    val_loader   = make_loader(data_dir, "val",   batch_size, use_resnet=True, shuffle=False)

    best_acc = 0.0
    patience_counter = 0
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # ============================================================
    #  VÒNG LẶP HUẤN LUYỆN
    # ============================================================
    for epoch in range(1, epochs + 1):

        # train 1 epoch
        tr_loss, tr_acc = train_epoch(model, train_loader, device, criterion, optimizer)
        # validate
        val_loss, val_acc = eval_epoch(model, val_loader, device, criterion)

        print(f"Epoch {epoch:02d} | Train {tr_acc*100:.2f}% | Val {val_acc*100:.2f}% | "
              f"Loss T/V {tr_loss:.4f}/{val_loss:.4f}")

        scheduler.step(val_loss)  # điều chỉnh learning rate

        # --------------------------------------------------------
        #  Lưu model tốt nhất theo validation accuracy
        # --------------------------------------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model to {model_path} (val_acc={best_acc*100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

            # ----------------------------------------------------
            #  EARLY STOPPING
            # ----------------------------------------------------
            if patience_counter >= patience:
                print("⏹ Early stopping triggered!")
                print(f"Epoch hiện tại: {epoch}")
                print(f"Số epoch không cải thiện liên tiếp: {patience_counter}")
                print(f"Best validation accuracy đạt được: {best_acc * 100:.2f}%")
                print(f"Train acc cuối: {tr_acc * 100:.2f}% | Val acc cuối: {val_acc * 100:.2f}%")
                print(f"Train loss cuối: {tr_loss:.4f} | Val loss cuối: {val_loss:.4f}")
                break

    print("🎉 Training finished.")
