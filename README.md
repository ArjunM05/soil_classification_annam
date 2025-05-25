# Soil Classification Challenges: Multiclass and Binary Classification

This repository contains the code and resources for two distinct Kaggle challenges related to soil image classification:

- **Part 1: Multiclass Classification** – Distinguishing between Alluvial, Red, Black, and Clay soil types.
- **Part 2: Binary Classification** – Identifying whether an image contains soil or not.

The project aims to demonstrate robust deep learning solutions for automated soil classification.

## Table of Contents

1.  [Challenge 1: Multiclass Classification](#challenge-1-multiclass-classification)
    - [Goal](#goal-1)
    - [Team](#team)
    - [Approach](#approach-1)
    - [Evaluation](#evaluation-1)
    - [Files](#files-1)
2.  [Challenge 2: Binary Classification](#challenge-2-binary-classification)
    - [Goal](#goal-2)
    - [Team](#team-2)
    - [Approach](#approach-2)
    - [Evaluation](#evaluation-2)
    - [Files](#files-2)
3.  [Setup Instructions](#setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Data Download](#data-download)
    - [Folder Structure](#folder-structure)
    - [Installation](#installation)
4.  [Run Instructions](#run-instructions)
    - [Running Challenge 1 (Multiclass)](#running-challenge-1-multiclass)
    - [Running Challenge 2 (Binary)](#running-challenge-2-binary)

---

## Challenge 1: Multiclass Classification

This part of the project focuses on classifying soil images into four distinct categories: Alluvial, Red, Black, and Clay soil.

### Goal

The primary goal is to develop a deep learning model capable of accurately identifying the type of soil from an image, submitting results to the "Soil Classification Part 1" Kaggle challenge.

### Team

- **Team Name**: Expendables
- **Team Leader**: Sushmetha S R
- **Team Members**: Abhinav Chaitanya R, Arjun M, Harshavardhan S, Kiranchandran H, Sushmetha S R
  - **Affiliation**: Vellore Institute of Technology, VIT Chennai

### Approach

- **Model**: An EfficientNetV2-S model, pretrained on ImageNet, was used as the base model for transfer learning.
- **Preprocessing**:
  - Images are resized to 256x256 and then center-cropped to 224x224 pixels.
  - Normalization is applied using ImageNet statistics.
  - Data augmentation techniques like random resized cropping and horizontal flips are used for the training set.
  - Categorical soil types are encoded into numerical labels using `LabelEncoder`.
- **Training**:
  - Trained for 30 epochs with early stopping.
  - Adam optimizer with a learning rate of 0.0001.
  - CrossEntropyLoss as the loss function.
- **Inference**:
  - The best model (`best_model.pth`) is loaded.
  - The same preprocessing pipeline is applied to test images.
  - The class with the highest predicted probability is selected as the final label.

### Evaluation

- **Metrics**: F1-score is the primary evaluation metric, calculated class-wise (Alluvial, Black, Clay, Red soil).
- **Visuals**: Includes image distribution, training history plots (loss and F1-score over epochs), and a confusion matrix to assess class-wise accuracy.
- **Leaderboard Rank**: 73

### Files

- `project-card.ipynb`: Jupyter notebook containing the project details, approach, and evaluation for Multiclass Classification.
- `preprocessing.py`: Python script containing reusable preprocessing functions and the `SoilDataset` class.
- `training.ipynb`: The primary training notebook for Multiclass Classification. (Note: The provided `training.ipynb` in the current context was for Binary Classification, but based on the `project-card.ipynb` it seems there's an analogous `training.ipynb` for multiclass as well).
- `inference.ipynb`: Jupyter notebook for generating predictions on the test set.
- `ml-metrics.json`: JSON file containing the evaluation metrics from the model training.
- `training_history.png`: Visualization of training loss and validation F1-score over epochs.
- `confusion_matrix.png`: Visual confusion matrix of model performance.

---

## Challenge 2: Binary Classification

This part of the project addresses the binary classification task: determining if an image contains soil or not.

### Goal

The goal is to accurately classify images as 'soil' (positive) or 'non-soil' (negative) and achieve a high F1-score for the binary classification challenge.

### Team

- **Author**: Sushmetha S R
- **Team Name**: expendables
- **Team Members**: Abhinav Chaitanya R, Arjun M, Harshavardhan S, Kiranchandran H, Sushmetha S R
- **Leaderboard Rank**: 25

### Approach

- **Model**: EfficientNet-B0, pretrained on ImageNet, is used as a feature extractor. The classifier head is replaced with an Identity layer to obtain 1280-D features.
- **Preprocessing**:
  - Images are resized to 224x224.
  - Converted to PyTorch tensors and normalized using ImageNet statistics.
  - A custom `SoilDataset` class handles robust image loading, skipping invalid images.
- **Classification Method**:
  - Features are extracted from both training and validation sets.
  - Cosine similarity is calculated between validation features and training features.
  - An optimal threshold for classification is determined based on the F1-score on the validation set.
  - Images with similarity scores above the threshold are classified as 'soil'.

### Evaluation

- **Metrics**: F1-score, Precision, Recall, and Accuracy for binary classification.
- **Visuals**: Includes display of sample training images and analysis of PCA feature distribution and RGB distribution.
- **ml-metrics.json**: Contains the final binary classification metrics.

### Files

- `training.ipynb`: Jupyter notebook containing the full code for binary classification model training, feature extraction, similarity calculation, and evaluation.
- `ml-metrics.json`: JSON file storing the binary classification metrics.

---

## Setup Instructions

### Prerequisites

Ensure you have Python installed (version 3.10+ recommended).
The following Python packages are required:

Pillow>=9.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
torch>=1.10.0
torchvision>=0.11.0
tqdm>=4.62.0

### Data Download

The datasets for both challenges are part of Kaggle competitions. You will need to download them from the respective Kaggle competition pages:

- **Challenge 1 (Multiclass Classification)**: Check the Kaggle competition page associated with "Soil Classification Part 1".
- **Challenge 2 (Binary Classification)**: Check the Kaggle competition page associated with "Soil Classification Part 2". The `training.ipynb` for Part 2 references data at `/kaggle/input/soil-classification-part-2/soil_competition-2025/`.

### Folder Structure

It is recommended to structure your project directory as follows, especially for the binary classification challenge data paths mentioned in `training.ipynb`:

.
├── requirements.txt
├── preprocessing.py (for multiclass preprocessing)
├── training.ipynb (for Binary Classification)
├── project-card.ipynb (info for Multiclass)
├── project-card-1.ipynb (info for Multiclass)
├── data/
│ ├── soil_classification-2025/ (for Challenge 1 - Multiclass)
│ │ ├── train/
│ │ ├── test/
│ │ └── train_labels.csv
│ │ └── test_ids.csv
│ └── soil_competition-2025/ (for Challenge 2 - Binary)
│ ├── train/ (contains soil images)
│ └── train_labels.csv (contains image_ids for binary challenge)
└── models/
└── best_model.pth (trained model for Multiclass, if generated)
└── outputs/
├── ml-metrics.json
├── training_history.png (Multiclass)
└── confusion_matrix.png (Multiclass)

**Note on data paths:** The notebooks are written with Kaggle environment paths (e.g., `/kaggle/input/...`). You might need to adjust these paths in the Python scripts/notebooks (`training.ipynb`, `preprocessing.py`) to match your local setup after downloading the data.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Run Instructions

### Running Challenge 1 (Multiclass)

1.  **Preprocessing**: The `preprocessing.py` script likely contains the `SoilDataset` class and transformation definitions. You would typically import these into your training notebook.
2.  **Training**: Open and run the `training.ipynb` (for multiclass) notebook in a Jupyter environment. This notebook will handle data loading, model training, and evaluation.
3.  **Inference**: After training, use `inference.ipynb` to generate predictions on the test set and create a submission file.

    ```bash
    # Example (assuming you have jupyter installed and are in the repo root)
    jupyter notebook training.ipynb # For multiclass training
    jupyter notebook inference.ipynb # For multiclass inference
    ```

### Running Challenge 2 (Binary)

1.  **Training and Evaluation**: Open and run the `training.ipynb` (for binary classification) notebook. This notebook covers all steps from data loading and feature extraction to cosine similarity calculation, optimal threshold finding, and evaluation.

    ```bash
    # Example (assuming you have jupyter installed and are in the repo root)
    jupyter notebook training.ipynb # For binary classification
    ```

    This will generate `ml-metrics.json` with the binary classification results.

---
