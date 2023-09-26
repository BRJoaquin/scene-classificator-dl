# Deep Learning for Scene Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Implementation Details](#implementation-details)
4. [Model Architectures](#model-architectures)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Environment Setup](#environment-setup)
7. [Tools Used](#tools-used)
8. [How to Run](#how-to-run)

## Introduction

This project aims to apply Deep Learning techniques to a real-world use case of scene classification. The focus will be on assessing the performance of various models in classifying different scenes. The project is structured to compare a simple baseline model with advanced architectures like ResNet and DenseNet.

## Dataset

The dataset comprises over 13,000 images grouped into 6 categories: Buildings, Forests, Glaciers, Mountains, Oceans, and Streets. The dataset is included in the repository under the `/data` folder.

The dataset is organized into two folders:
- `train_set`: Contains the training images, further grouped into sub-folders by category.
- `test_set`: Contains the test images, also grouped into sub-folders by category.

## Implementation Details

- The project will be implemented locally.
- Models will be built from scratch, tailored to fit this specific dataset.
    - Pre-trained models are not under consideration for this project.
- Decisions regarding image pre-processing (transforms, augmentation, etc.) will be based on a thorough exploration of the dataset.
- All codes will be encapsulated in a Jupyter Notebook `.ipynb` file for easier review and reproduction.

## Model Architectures

Three different models will be explored:
1. **SimpleCNN**: A baseline architecture designed to provide a performance baseline. Consists of three convolutional layers and two fully connected layers.
2. **ResNet**: A popular architecture known for its residual learning technique, which enables training deeper networks.
3. **DenseNet**: An architecture that employs dense connections between layers, thus improving gradient flow and feature reuse.

## Evaluation Metrics

The models' performances will be evaluated based on the following metrics, computed using the `sklearn` library:
1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. Confusion Matrix

The project will also include visualizations and logs to track the model's performance during training and validation phases. TensorBoard will be used for monitoring.

## Environment Setup

- Conda will be used for package management and to set up the environment.
- A `requirements.txt` file will be included to reproduce the environment.

## Tools Used

- [Jupyter Notebook](https://jupyter.org/)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Conda](https://docs.conda.io/en/latest/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

## How to Run

Instructions for setting up the environment and running the notebook will be provided here.

