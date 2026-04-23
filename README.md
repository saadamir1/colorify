# Colorify 🎨

A deep learning project for **automatic image colorization** — converting grayscale images to color using multiple neural network architectures, trained and evaluated on Kaggle.

---

## What This Project Does

Given a grayscale image as input, the models learn to predict and output a plausible colorized (RGB) version of that image. Three different architectures were explored and compared:

| Notebook | Architecture | Approach |
|---|---|---|
| `unet-model.ipynb` | U-Net | Encoder-decoder with skip connections |
| `ResNet.ipynb` | ResNet-34 Autoencoder | Pretrained ResNet-34 encoder + custom decoder |
| `base-gan-pix2pix.ipynb` | Pix2Pix GAN | FCN-ResNet50 generator + CNN discriminator |
| `pix2pix-without-finetune.ipynb` | Pix2Pix (no fine-tune) | Pix2Pix without pretrained weights |
| `resnet-without-finetuning.ipynb` | ResNet (no fine-tune) | ResNet autoencoder without pretrained weights |
| `dataloader_code.ipynb` | — | Shared dataset/dataloader utilities |

---

## Dataset

The project uses a custom dataset (`genai-colorify-dataset-version-1`) hosted on Kaggle, structured as:

```
dataset/
├── train/
│   ├── gray/      ← grayscale input images
│   └── color/     ← ground-truth RGB images
└── test/
    ├── gray/
    └── color/
```

Each grayscale image has a corresponding color image with the same filename.

---

## Models

### 1. U-Net (`unet-model.ipynb`)
- Classic encoder-decoder architecture with skip connections
- Input: `[B, 1, 256, 256]` grayscale
- Output: `[B, 3, 256, 256]` RGB
- Uses `segmentation-models-pytorch` library

### 2. ResNet-34 Autoencoder (`ResNet.ipynb`)
- Encoder: Pretrained ResNet-34 (layers up to last conv block)
- Decoder: Custom upsampling layers (bilinear interpolation + Conv2d)
- Loss: MSE (L2)
- Optimizer: Adam (lr=1e-4), StepLR scheduler
- Trained for 30 epochs, batch size 128

### 3. Pix2Pix GAN (`base-gan-pix2pix.ipynb`)
- Generator: FCN-ResNet50 (pretrained, modified for 1-channel input and 3-channel output)
- Discriminator: Simple CNN with LeakyReLU activations
- Loss: Adversarial (BCE) + Pixel-wise (L1)
- Optimizer: Adam (lr=0.0002, β=(0.5, 0.999))
- Trained for 20 epochs, batch size 16

---

## Data Augmentation

Applied consistently across all models:

**Grayscale transforms:**
- `RandomResizedCrop(256×256, scale=(0.8, 1.0))`
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(10°)`
- `Normalize(mean=0.5, std=0.5)`

**Color transforms** (same as above, plus):
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`
- `Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))`

---

## Requirements

```
torch
torchvision
segmentation-models-pytorch
Pillow
matplotlib
tqdm
```

Install via:
```bash
pip install torch torchvision segmentation-models-pytorch Pillow matplotlib tqdm
```

---

## Running on Kaggle

All notebooks are designed to run on Kaggle with GPU acceleration. Dataset paths are set to `/kaggle/input/...`. To run locally, update the `root_dir` paths in the dataloader cells.

---

## Results

| Model | Epochs | Final Loss |
|---|---|---|
| ResNet-34 Autoencoder | 30 | ~0.163 (MSE) |
| Pix2Pix GAN | 20 | G: ~100, D: ~0.03 |

Loss curves are plotted at the end of each notebook.

---

## Saved Models

Each notebook saves the trained model weights:
- `ResNet.ipynb` → `vgg16_colorization.pth`
- `base-gan-pix2pix.ipynb` → `Pix2Pix_Generator.pth`, `Pix2Pix_Discriminator.pth`
