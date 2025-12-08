
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.models.resnet import EmotionResNet
from src.utils.dataset import make_loader, EMOTIONS

def evaluate(data_dir="data", model_path="models/emotion_resnet18.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionResNet(num_classes=len(EMOTIONS)).to(device)

    # Load checkpoint
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    loader = make_loader(data_dir, "val", batch_size=64, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

    # Classification report
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    # === Biểu đồ ===
    # 1. Confusion matrix heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # 2. Precision, Recall, F1-score theo từng lớp
    metrics = ["precision", "recall", "f1-score"]
    for metric in metrics:
        values = [report[label][metric] for label in EMOTIONS]
        plt.figure(figsize=(8,6))
        sns.barplot(x=EMOTIONS, y=values, palette="viridis")
        plt.ylabel(metric.capitalize())
        plt.title(f"Per-Class {metric.capitalize()}")
        plt.show()

def plot_training(history):
    """
    Vẽ Loss và Accuracy theo epoch
    history: dict chứa train_loss, val_loss, train_acc, val_acc
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10,4))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    plt.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
