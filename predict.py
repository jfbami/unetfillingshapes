import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from model import DeepUNet


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
