# Task-1
# Simple Emotion Recognition using CNN

This repository contains code for a simple emotion recognition model using Convolutional Neural Networks (CNN) implemented in PyTorch. The model is trained and tested on a dataset containing facial expressions of different emotions.

## Dataset

The code uses the [Kaggle Facial Expression Recognition Challenge dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). The dataset is organized into training and testing sets, each containing images of faces labeled with different emotions.

### Dataset Structure

- train: Contains training images grouped by emotion classes.
- test: Contains testing images without emotion labels.

## Model Architecture

The CNN model architecture is defined in the SimpleEmotionCNN class in the main script (emotion_cnn.py). The model consists of two convolutional layers, max-pooling layers, and fully connected layers.

## Usage

1. *Dependencies:*
   - Python 3.x
   - PyTorch
   - torchvision
   - scikit-learn
   - matplotlib
   - seaborn
   - tqdm
   - Pillow

2. *Install Dependencies:*
   ```bash
   pip install torch torchvision scikit-learn matplotlib seaborn tqdm Pillow
