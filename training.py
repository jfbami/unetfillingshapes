import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from model import DeepUNet


class GetDataset(Dataset):
    def __init__(self, outlines_dir, filled_dir):
        self.outline_dir = outlines_dir
        self.filled_dir = filled_dir
        self.images = sorted([f for f in os.listdir(outlines_dir) if f.endswith('.png')])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx]
        o = cv2.imread(os.path.join(self.outline_dir, file_name), 0) / 255.0
        f = cv2.imread(os.path.join(self.filled_dir, file_name), 0) / 255.0
        return torch.Tensor(o).unsqueeze(0), torch.Tensor(f).unsqueeze(0)


def compute_iou(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    if union == 0:
        return 1.0
    return (intersection / union).item()


def train_model(epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GetDataset("data/outlines", "data/filled")
    train_ds, val_ds = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = DeepUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        t_iou = 0.0
        for out, fill in train_loader:
            out, fill = out.to(device), fill.to(device)
            optimizer.zero_grad()
            output = model(out)
            loss = criterion(output, fill)
            loss.backward(); optimizer.step()
            t_loss += loss.item()
            t_iou += compute_iou(output, fill)

        model.eval()
        v_loss = 0.0
        v_iou = 0.0
        with torch.no_grad():
            for o, f in val_loader:
                o, f = o.to(device), f.to(device)
                output = model(o)
                v_loss += criterion(output, f).item()
                v_iou += compute_iou(output, f)

        avg_t_iou = t_iou / len(train_loader)
        avg_v_iou = v_iou / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {t_loss/len(train_loader):.4f} | Train IoU: {avg_t_iou:.4f} | "
              f"Val Loss: {v_loss/len(val_loader):.4f} | Val IoU: {avg_v_iou:.4f}")

    torch.save(model.state_dict(), "shape_filler.pth")
    print(f"\nFinal Val IoU: {avg_v_iou:.4f}")

if __name__ == "__main__":
    train_model(epochs=15)
