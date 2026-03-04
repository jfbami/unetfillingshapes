
#INTRO!!!
A deep learning model that detects incomplete or outlined shapes in images and fills them with solid regions.

---

##How It Works

The model takes a black and white image containing a shape outline (or partial shape) and outputs the same image with the shape fully filled in. It handles a variety of shape types — rectangles, circles, ellipses, curves, and more at different scales and positions.

---

## Input → Output

<img width="1603" height="444" alt="image" src="https://github.com/user-attachments/assets/aa916bb3-13f4-41e3-aca0-894934d097cd" />


| Shape Type | Input | Output |
|---|---|---|
| Rectangle | White outline on black background | Solid filled rectangle |
| Circle | Thin circular outline | Solid filled circle |
| Ellipse | Partial/tilted ellipse outline | Solid filled ellipse |
| Curve / Arc | Partial curved stroke | Reconstructed arc |
| Point | Single pixel | Preserved dot |

---

##  Architecture

Built on a **Deep U-Net (DeepUNet)** architecture — an encoder-decode CNN

- **Encoder**: Successive downsampling blocks that extract shape features
- **Bottleneck**: Dense feature representation
- **Decoder**: Upsampling blocks with skip connections from the encoder
- **Output**: Single-channel binary mask (filled shape)
---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-scientific-lightblue?logo=numpy)

---

##  Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/jfbaami/AI-Challenge---Shape-Filling-AI-.git
cd AI-Challenge---Shape-Filling-AI-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── training.py         # Model training script
├── predict.py          # Inference script
├── model.py            # DeepUNet architecture definition
├── dataset.py          # Data loading and augmentation
├── requirements.txt    # Dependencies
└── README.md
```

---
