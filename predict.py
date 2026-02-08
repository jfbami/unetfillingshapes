import cv2
import torch
import numpy as np
import os

# 1. Setup Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepUNet().to(device)
if os.path.exists("shape_filler.pth"):
    model.load_state_dict(torch.load("shape_filler.pth", map_location=device))
model.eval()
input_folder = "data/outlines"
files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

print("use any key to move to the next image or Press'q to quit.")

for file_name in files:
    img_path = os.path.join(input_folder, file_name)
    original = cv2.imread(img_path, 0)
    input_tensor = torch.Tensor(original).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    #predict
    with torch.no_grad():
        prediction = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

    #postprocess for Display (Convert back to 0-255 ints)
    predicted_img = (prediction * 255).astype(np.uint8)

    #stack images horizontally
    combined_view = np.hstack((original, predicted_img))
    cv2.imshow("Shape Filling AI", combined_view)

    key = cv2.waitKey(0)
    if key == ord('q'):  # Allow quitting early
        break

cv2.destroyAllWindows()