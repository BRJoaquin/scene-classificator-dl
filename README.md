# Deep Learning for Scene Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Implementation Details](#implementation-details)
4. [Model Architectures](#model-architectures)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Performance Overview](#performance-overview)
7. [Environment Setup](#environment-setup)
8. [Tools Used](#tools-used)
9. [How to Run](#how-to-run)
10. [Pre-trained Models](#pre-trained-models)
11. [Final Conclusion](#final-conclusion)

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

## Performance Overview

Here are the summarized metrics for the models:

| Metric     | Simple CNN | ResNet | DenseNet |
|------------|------------|--------|----------|
| Accuracy   | 83.60%     | 88.23% | 85.80%   |
| Precision  | 83.85%     | 88.41% | 86.34%   |
| Recall     | 83.80%     | 88.44% | 85.92%   |
| F1 Score   | 83.77%     | 88.34% | 85.84%   |

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

## Pre-trained Models

If you wish to skip the training process, pre-trained models are available for each architecture. You can find them as follows:
- `best_densenet_model.pth`: Pre-trained DenseNet model.
- `best_resnet_model.pth`: Pre-trained ResNet model.
- `best_simple_cnn_model.pth`: Pre-trained SimpleCNN model.

To use these models, simply load them using PyTorch's `torch.load()` function.

## Final Conclusion

Through extensive experimentation with different neural network architectures, we've obtained valuable insights into scene classification. Despite variances in model complexity and performance metrics, each of the three examined models—SimpleCNN, ResNet, and DenseNet—proved competent in solving the task at hand, albeit with varying degrees of success. Notably, we found some classes that were consistently misclassified across all models, indicating room for further investigation. It appears that the challenges here might lie within the dataset itself rather than the classifiers. We invite readers to delve into the specifics and draw their own conclusions.