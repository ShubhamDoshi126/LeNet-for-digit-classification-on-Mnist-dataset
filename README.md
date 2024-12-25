# LeNet for MNIST Digit Classification

A comprehensive implementation of LeNet architecture for handwritten digit recognition using the MNIST dataset.

## Overview

This project implements a Convolutional Neural Network (CNN) based on LeNet architecture for classifying handwritten digits. The system uses TensorFlow and Keras to train, evaluate, and make predictions on digit images.

## Project Structure

```
.
├── module8.ipynb       # Jupyter notebook for model training
├── module8.py         # Script for image prediction
└── le_net.py         # LeNet architecture implementation
```

## Features

**Model Architecture**
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- Sequential API implementation

**Training Capabilities**
- MNIST dataset integration
- Model compilation with optimizer and loss function
- Performance evaluation metrics
- Model checkpointing

**Image Processing**
- Input image preprocessing
- Normalization and reshaping
- Size compatibility handling

**Model Management**
- Save trained models
- Load existing models
- Prediction interface

## Usage

**Training the Model**
```python
from le_net import LeNet

# Initialize and train
lenet = LeNet(batch_size=128, epochs=10)
lenet.train()

# Save model
lenet.save("doshi_cnn_model")
```

**Making Predictions**
```bash
python module8.py <image_filename> <expected_digit>
```

Example:
```bash
python module8.py 0_0.png 0
```

## Technical Implementation

**CNN Architecture**
- Input Layer: 28x28 grayscale images
- Convolutional Layers: Feature extraction
- Pooling Layers: Spatial reduction
- Dense Layers: Classification
- Output: 10 classes (digits 0-9)

**Training Parameters**
- Batch Size: 128
- Epochs: 10
- Optimizer: Adam
- Loss Function: Categorical Cross-entropy

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Python 3.x
