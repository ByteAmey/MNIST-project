# MNIST Handwritten Digit Classification with PyTorch

A beginner-friendly project that classifies handwritten digits (0–9) from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using PyTorch. It includes two models so you can compare approaches:

1. **A fully connected neural network (MLP)**
2. **A simple Convolutional Neural Network (CNN)**

## Results

| Model | Notebook | Epochs | Test Accuracy | Saved Weights |
|-------|----------|--------|---------------|---------------|
| MLP (`DigitModel`) | `mnist_project.ipynb` | 5 | **97.51%** | `mnist_model.pth` |
| CNN (`SimpleCNN`) | `CNN_project.ipynb` | 3 | **98.32%** | `simple_cnn_mnist.pth` |

The CNN reaches higher accuracy in fewer epochs because convolutional layers can pick up spatial patterns (strokes, curves, edges) that a flattened MLP input cannot.

## Project Structure

```
MNIST-project/
├── data/                    # MNIST dataset (downloaded automatically)
├── mnist_project.ipynb      # MLP model + sample prediction visualisation
├── CNN_project.ipynb        # CNN model training and evaluation
├── mnist_model.pth          # Trained MLP weights
├── simple_cnn_mnist.pth     # Trained CNN weights
├── .gitignore
└── README.md
```

## Model Architectures

### 1. MLP (`DigitModel`)

```
Input (28×28 → 784)
  → Linear(784, 128) → ReLU
  → Linear(128, 64)  → ReLU
  → Linear(64, 10)
```

### 2. CNN (`SimpleCNN`)

```
Input (1×28×28)
  → Conv2d(1, 8, 3, padding=1)  → ReLU → MaxPool(2)   # 8×14×14
  → Conv2d(8, 16, 3, padding=1) → ReLU → MaxPool(2)   # 16×7×7
  → Flatten (784)
  → Linear(784, 64) → ReLU
  → Linear(64, 10)
```

## Training Setup

| Setting | Value |
|---------|-------|
| Loss function | Cross-Entropy Loss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Preprocessing | `ToTensor()` (pixel values scaled to 0–1) |
| Device | Automatically uses GPU (CUDA) if available, otherwise CPU |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/MNIST-project.git
cd MNIST-project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision matplotlib jupyter
```

### 4. Run the notebooks

```bash
jupyter notebook
```

Open either `mnist_project.ipynb` (MLP) or `CNN_project.ipynb` (CNN) and run all cells. The MNIST dataset will be downloaded automatically into the `data/` folder on first run.

## Loading a Saved Model

```python
import torch

# Define the same architecture used in training (e.g. SimpleCNN), then:
model = SimpleCNN()
model.load_state_dict(torch.load("simple_cnn_mnist.pth", map_location="cpu"))
model.eval()
```

## Sample Prediction

The MLP notebook also picks an image from the test set, displays it, and compares the model's prediction with the true label:

```
User picked image index 100
Actual Label 6
Model Prediction 6
```

## Tech Stack

- Python
- PyTorch
- Torchvision
- Matplotlib
- Jupyter Notebook

## Possible Improvements

- Train for more epochs and add a learning-rate scheduler
- Add dropout / batch normalisation to reduce overfitting
- Plot training loss and accuracy curves
- Add data augmentation (rotation, shifts)
- Build a small app that lets you draw a digit and predict it
