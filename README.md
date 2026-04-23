# Colorify 🎨

A deep learning project for **automatic image colorization** — converting grayscale images to color using multiple neural network architectures, trained and evaluated on Kaggle.

---

## Table of Contents
- [What This Project Does](#what-this-project-does)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Running the Notebooks](#running-the-notebooks)
  - [On Kaggle (Recommended)](#on-kaggle-recommended)
  - [Locally](#locally)
- [Models](#models)
- [Data Augmentation](#data-augmentation)
- [Results](#results)
- [Saved Models](#saved-models)

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

## Project Structure

```
colorify/
├── unet-model.ipynb               ← U-Net colorization
├── ResNet.ipynb                   ← ResNet-34 autoencoder colorization
├── base-gan-pix2pix.ipynb         ← Pix2Pix GAN (with pretrained weights)
├── pix2pix-without-finetune.ipynb ← Pix2Pix GAN (no pretrained weights)
├── resnet-without-finetuning.ipynb← ResNet autoencoder (no pretrained weights)
└── dataloader_code.ipynb          ← Shared dataloader utilities
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (strongly recommended; CPU training will be very slow)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/colorify.git
cd colorify
```

### 2. Install dependencies
```bash
pip install torch torchvision segmentation-models-pytorch Pillow matplotlib tqdm
```

> **Note:** For a specific CUDA version of PyTorch, visit [pytorch.org](https://pytorch.org/get-started/locally/) and use the appropriate install command, e.g.:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 3. Install Jupyter (if running locally)
```bash
pip install notebook
```

---

## Dataset

The project uses a custom dataset (`genai-colorify-dataset-version-1`) hosted on Kaggle.

**Dataset structure expected:**
```
dataset/
├── train/
│   ├── gray/      ← grayscale input images
│   └── color/     ← ground-truth RGB images
└── test/
    ├── gray/
    └── color/
```

Each grayscale image must have a corresponding color image with the **same filename**.

**To download the dataset from Kaggle:**
1. Install the Kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` API token in `~/.kaggle/`
3. Run:
```bash
kaggle datasets download -d <dataset-slug>
unzip <dataset-slug>.zip -d dataset/
```

---

## Running the Notebooks

### On Kaggle (Recommended)

All notebooks are designed to run on Kaggle with GPU acceleration — no local setup needed.

1. Go to [kaggle.com](https://www.kaggle.com) and create an account
2. Upload or fork the notebook
3. Attach the dataset `genai-colorify-dataset-version-1` under **Add Data**
4. Enable GPU: **Settings → Accelerator → GPU**
5. Click **Run All**

Dataset paths in the notebooks are already set to `/kaggle/input/genai-colorify-dataset-version-1/`.

---

### Locally

1. Complete the [Setup & Installation](#setup--installation) steps above
2. Download the dataset and place it somewhere on your machine
3. Open a notebook:
```bash
jupyter notebook ResNet.ipynb
```
4. **Update the dataset paths** in the dataloader cell. Find lines like:
```python
root_dir="/kaggle/input/genai-colorify-dataset-version-1/train"
```
and change them to your local path, e.g.:
```python
root_dir="./dataset/train"
```
5. Run all cells top to bottom

> **Tip:** Start with `ResNet.ipynb` — it's the simplest model and trains fastest.

---

## Models

### 1. U-Net (`unet-model.ipynb`)
- Classic encoder-decoder architecture with skip connections
- Input: `[B, 1, 256, 256]` grayscale → Output: `[B, 3, 256, 256]` RGB
- Uses `segmentation-models-pytorch` library

### 2. ResNet-34 Autoencoder (`ResNet.ipynb`)
- Encoder: Pretrained ResNet-34 (all layers up to the last conv block)
- Decoder: Custom upsampling layers (bilinear interpolation + Conv2d)
- Loss: MSE (L2) | Optimizer: Adam (lr=1e-4) | Scheduler: StepLR
- 30 epochs, batch size 128

### 3. Pix2Pix GAN (`base-gan-pix2pix.ipynb`)
- Generator: FCN-ResNet50 (pretrained, modified for 1-channel input and 3-channel output)
- Discriminator: Simple CNN with LeakyReLU activations
- Loss: Adversarial (BCE) + Pixel-wise (L1)
- Optimizer: Adam (lr=0.0002, β=(0.5, 0.999))
- 20 epochs, batch size 16

### 4. Ablation Variants
- `resnet-without-finetuning.ipynb` — same as ResNet-34 but with randomly initialized weights
- `pix2pix-without-finetune.ipynb` — same as Pix2Pix but with randomly initialized weights

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

## Results

| Model | Epochs | Final Loss |
|---|---|---|
| ResNet-34 Autoencoder | 30 | ~0.163 (MSE) |
| Pix2Pix GAN | 20 | G: ~100, D: ~0.03 |

Loss curves are plotted at the end of each notebook.

---

## Saved Models

Each notebook saves the trained model weights after training:

| Notebook | Saved File(s) |
|---|---|
| `ResNet.ipynb` | `vgg16_colorization.pth` |
| `base-gan-pix2pix.ipynb` | `Pix2Pix_Generator.pth`, `Pix2Pix_Discriminator.pth` |
