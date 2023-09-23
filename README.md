# Deep Learning for Scene Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Implementation Details](#implementation-details)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Environment Setup](#environment-setup)
6. [Tools Used](#tools-used)

## Introduction

This project aims to apply Deep Learning techniques to a real-world use case of scene classification. The focus will be on assessing the performance of various models in classifying different scenes.

## Dataset

The dataset comprises over 13,000 images grouped into 6 categories: Buildings, Forests, Glaciers, Mountains, Oceans, and Streets. The dataset it's included in the respository under `/data` folder.

The dataset is organized into two folders:
- `train_set`: Contains the training images, further grouped into sub-folders by category.
- `test_set`: Contains the test images, also grouped into sub-folders by category.

## Implementation Details

- The project will be implemented locally.
- Models will be built from scratch, tailored to fit this specific dataset.
    - Pre-trained models are not under consideration for this project.
- Decisions regarding image pre-processing (transforms, augmentation, etc.) will be based on a thorough exploration of the dataset.
- All codes will be encapsulated in a Jupyter Notebook `.ipynb` file for easier review and reproduction.

## Evaluation Metrics

The models' performances will be evaluated based on the following metrics, computed using the `sklearn` library:
1. Accuracy
2. Precision
3. Recall
4. F1 Score

The project will also include visualizations and logs to track the model's performance during training and validation.

## Environment Setup

- Conda will be used for package management and to set up the environment.
- A `requirements.txt` file will be included to reproduce the environment.

## Tools Used

- [Jupyter Notebook](https://jupyter.org/)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Conda](https://docs.conda.io/en/latest/)

