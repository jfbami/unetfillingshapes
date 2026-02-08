import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

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

class DeepUNet(nn.Module):
    def __init__(self):
        super(DeepUNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)
            )
        self.pool = nn.MaxPool2d(2, 2)
        
        
        #4 Levels to reach our 32x32 bottleneck
        self.enc1 = conv_block(1, 32); self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128); self.enc4 = conv_block(128, 256)
        self.bottleneck = conv_block(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec4 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2); self.dec1 = conv_block(64, 32)
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x); p1 = self.pool(s1); s2 = self.enc2(p1); p2 = self.pool(s2)
        s3 = self.enc3(p2); p3 = self.pool(s3); s4 = self.enc4(p3); p4 = self.pool(s4)
        b = self.bottleneck(p4)
        d4 = self.up4(b); d4 = torch.cat((d4, s4), dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat((d3, s3), dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat((d2, s2), dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat((d1, s1), dim=1); d1 = self.dec1(d1)
        return self.final_conv(d1)

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
        for out, fill in train_loader:
            out, fill = out.to(device), fill.to(device)
            optimizer.zero_grad()
            loss = criterion(model(out), fill)
            loss.backward(); optimizer.step(); t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for o, f in val_loader:
                v_loss += criterion(model(o.to(device)), f.to(device)).item()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss/len(train_loader):.4f} | Val Loss: {v_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), "shape_filler.pth")

if __name__ == "__main__":
    train_model(epochs=15)