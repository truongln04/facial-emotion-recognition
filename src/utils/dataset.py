# -*- coding: utf-8 -*-
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms

EMOTIONS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
EMO2IDX = {e:i for i,e in enumerate(EMOTIONS)}

class EmotionDataset(Dataset):
    def __init__(self, root, split="train", use_resnet=False):
        """
        root: thư mục gốc data/
        split: "train" hoặc "val"
        use_resnet: True nếu dùng ResNet/MobileNet/DenseNet pretrained (RGB 224x224)
                    False nếu dùng CNN tự viết (grayscale 48x48)
        """
        self.paths = []
        split_dir = os.path.join(root, "processed", split)
        for label in EMOTIONS:
            cls_dir = os.path.join(split_dir, label)
            if not os.path.isdir(cls_dir): continue
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith((".jpg",".jpeg",".png")):
                    self.paths.append((os.path.join(cls_dir, fn), label))

        # transform cho ResNet/MobileNet (RGB 224x224)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if split == "train" else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if split == "train" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        img = Image.open(path).convert("RGB")  # đọc bằng PIL
        x = self.transform(img)
        y = torch.tensor(EMO2IDX[label]).long()
        return x, y

def make_loader(root, split, batch_size, use_resnet=False, shuffle=True):
    ds = EmotionDataset(root, split, use_resnet=use_resnet)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
