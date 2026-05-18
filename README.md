
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

Built on a DeepUNet architecture (CNN)

- **Encoder**: Successive downsampling blocks that extract shape features
- **Bottleneck**: Dense feature representation
- **Decoder**: Upsampling blocks with skip connections from the encoder
- **Output**: Single-channel binary mask (filled shape)
---


```
├── training.py         # Model training script
├── predict.py          # Inference script
├── model.py            # DeepUNet architecture definition
├── dataset.py          # Data loading and augmentation
├── requirements.txt    # Dependencies
└── README.md
```

---
