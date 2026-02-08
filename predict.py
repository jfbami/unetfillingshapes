import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt  # Added this missing import
import random

class DeepUNet(nn.Module):
    def __init__(self):
        super(DeepUNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)
            )
        self.pool = nn.MaxPool2d(2, 2)
        
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

#pred function
def predict_random_set(num_samples=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepUNet().to(device)

    # load the weights from training script
    if os.path.exists("shape_filler.pth"):
        model.load_state_dict(torch.load("shape_filler.pth", map_location=device))
        model.eval()
        print("Model loaded successfully.")
    else:
        print("Warning: 'shape_filler.pth' not found.")
        return

    input_folder = "data/outlines"
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    sample_files = random.sample(files, min(num_samples, len(files)))

    fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))

    for i, file_name in enumerate(sample_files):
        img_path = os.path.join(input_folder, file_name)
        original = cv2.imread(img_path, 0)
        
        input_tensor = torch.Tensor(original).unsqueeze(0).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            output = model(input_tensor)
            # Use Sigmoid to get pixel values between 0 and 1
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
   
        #top row
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f"Input {i+1}")
        axes[0, i].axis('off')

        #bottom row
        axes[1, i].imshow(pred, cmap='gray')
        axes[1, i].set_title(f"AI Output {i+1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_random_set(8)
