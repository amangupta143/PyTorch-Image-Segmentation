# PyTorch Image Segmentation Project

![PyTorch Image Segmentation](https://github.com/user-attachments/assets/12b9dd9f-d662-41d6-a03f-7acb32409556)


This repository contains an implementation of image segmentation using PyTorch and the U-Net architecture. The project focuses on human segmentation using the EfficientNet-B0 encoder and custom training pipeline.

## Overview

The project implements a complete image segmentation pipeline including:
- Custom dataset handling
- Data augmentation
- U-Net architecture with EfficientNet-B0 encoder
- Combined loss function (Dice + BCE)
- Training and validation loops
- Model checkpointing

## Requirements

```
torch
opencv-python
numpy
pandas
matplotlib
scikit-learn
tqdm
albumentations
segmentation-models-pytorch
```

## Project Structure

```
├── Dataset(folder)               # Training data information
├── Deep_Learning_with_PyTorch_ImageSegmentation.ipynb  # Main implementation file
└── helper.py                     # Helper functions for visualization
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/amangupta143/PyTorch-Image-Segmentation.git
cd PyTorch-Image-Segmentation
```

2. Install required packages:
```bash
pip install segmentation-models-pytorch
pip install -U albumentations
pip install opencv-contrib-python
```

3. Download the dataset:
```bash
git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git
```

## Model Architecture

- **Base Architecture**: U-Net
- **Encoder**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Channels**: 3 (RGB)
- **Output Classes**: 1 (Binary Segmentation)
- **Loss Function**: Combination of Dice Loss and Binary Cross-Entropy Loss

## Training Configuration

```python
EPOCHS = 35
LEARNING_RATE = 0.003
IMAGE_SIZE = 320
BATCH_SIZE = 16
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'
```

## Data Augmentation

Training augmentations include:
- Resize to 320x320
- Horizontal Flip (50% probability)
- Vertical Flip (50% probability)

## Model Training

The training process includes:
- Custom training and validation functions
- Model checkpointing for best validation loss
- Adam optimizer
- GPU acceleration support

```python
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

## Usage

1. Prepare your dataset and update the CSV_FILE path in the configuration
2. Run the training script:
```python
python deep_learning_with_pytorch_imagesegmentation.py
```

## Inference

The model can be used for inference as follows:

```python
model.load_state_dict(torch.load('bestModel.pt'))
image, mask = validset[idx]
logits_mask = model(image.to(DEVICE).unsqueeze(0))
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5) * 1.0
```

## Acknowledgments

Dataset originally from: [Human-Segmentation-Dataset](https://github.com/VikramShenoy97/Human-Segmentation-Dataset)

## License

MIT License

Feel free to use this implementation and modify it according to your needs. Contributions are welcome!
