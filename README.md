# Detection of Alzheimer's Disease using CNN

## Overview

This project implements a convolutional neural network (CNN) to detect Alzheimer's disease from EEG data. The model classifies subjects as cognitively normal (CN) or Alzheimer's disease (AD) using power spectral density features extracted from EEG signals.

## Features

- EEG data preprocessing with MNE-Python
- Power spectral density extraction using Welch's method
- CNN-based binary classification
- Hyperparameter tuning for learning rate, optimizer, and dropout
- Stratified train/validation/test split
- Model evaluation with accuracy, precision, recall, and F1-score

## Dataset

The dataset is sourced from OpenNeuro (dataset ID: ds004504). It contains EEG recordings from subjects classified as cognitively normal (CN) or Alzheimer's disease (AD).

**Classes:**
- CN (Cognitively Normal): Label 0
- AD (Alzheimer's Disease): Label 1

**Preprocessing:**
- EEG data loaded from `.set` files using MNE-Python
- 19 EEG channels selected: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2
- Resampled to 500 Hz
- 30-second epochs extracted (up to 20 epochs per subject)
- Power spectral density computed using Welch's method (frequency range: 4-40 Hz)
- Interpolated to 128 frequency bins and normalized per epoch
- Resized to 128×128 images

**Final dataset:**
- Total samples: 1,287
- CN samples: 580
- AD samples: 707
- Input shape: (128, 128, 1)

**Train/validation/test split:**
- Train: 900 samples (70%)
- Validation: 193 samples (15%)
- Test: 194 samples (15%)
- Stratified split to maintain class distribution

## Model Architecture

The CNN architecture consists of:

1. **Input Layer**: (128, 128, 1)
2. **Conv2D**: 16 filters, 3×3 kernel, padding='same'
3. **ReLU Activation**
4. **MaxPooling2D**: 2×2 pool size
5. **Conv2D**: 32 filters, 3×3 kernel, padding='same'
6. **ReLU Activation**
7. **MaxPooling2D**: 2×2 pool size
8. **Flatten Layer**
9. **Dense**: 64 units
10. **ReLU Activation**
11. **Dropout**: 0.25 (tuned)
12. **Output Dense**: 2 units with softmax activation

Total parameters: 2,102,146

## Technologies Used

- Python
- TensorFlow/Keras
- MNE-Python
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- openneuro-py

## Project Structure

```
.
├── datasetPreprocessing.ipynb    # Data download and preprocessing
├── hyperParamTuning.ipynb        # Model training and hyperparameter tuning
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Alzheimer_Disease_Detection_using_CNN
```

2. Install required dependencies:
```bash
pip install tensorflow mne pandas numpy scipy scikit-learn matplotlib openneuro-py
```

3. For Google Colab usage, install additional packages:
```bash
pip -q install openneuro-py mne pandas numpy scipy
```

## Usage

### 1. Dataset Preprocessing

Run `datasetPreprocessing.ipynb` to:
- Download the dataset from OpenNeuro (ds004504)
- Extract EEG features from `.set` files
- Compute power spectral density and generate 128×128 images
- Save processed data as `X.npy`, `y.npy`, and `subj.npy`

### 2. Model Training and Hyperparameter Tuning

Run `hyperParamTuning.ipynb` to:
- Load preprocessed data
- Split into train/validation/test sets
- Train baseline CNN model
- Tune hyperparameters (learning rate, optimizer, dropout)
- Train final model with best hyperparameters
- Evaluate on test set

### 3. Running Inference

To make predictions on new data:

```python
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('models/final_best.keras')

# Load and preprocess your EEG data (same pipeline as datasetPreprocessing.ipynb)
# X_preprocessed should have shape (n_samples, 128, 128, 1)

# Make predictions
predictions = model.predict(X_preprocessed)
predicted_classes = np.argmax(predictions, axis=1)
```

## Results

**Hyperparameter tuning results:**
- Best learning rate: 0.001
- Best optimizer: Adam
- Best dropout rate: 0.25 (0.35 also performed similarly)

**Final model test performance:**
- Test accuracy: 96.9%
- Test F1-score: 97.2%

Confusion matrix and training history are saved in the `results/` directory.

## Future Improvements

- Experiment with deeper CNN architectures (e.g., ResNet, VGG)
- Explore data augmentation techniques
- Test additional frequency bands and preprocessing methods
- Implement cross-validation for more robust evaluation
- Add support for multi-class classification if dataset includes other groups

## Author

Md. Fahim Karim
