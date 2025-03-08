# Image Segmentation

## Overview

This project implements an image segmentation model using the U-Net architecture on images from the dashboard of Self-driving cars. U-Net is a convolutional neural network (CNN) primarily used for biomedical image segmentation but can also be applied to various segmentation tasks.

## Features

- **Data Preprocessing**: Reads and processes images and masks for training.
- **U-Net Model Implementation**: Custom implementation of the U-Net architecture using TensorFlow/Keras.
- **Training and Evaluation**: Model training and visualization of segmentation results.
- **Prediction Visualization**: Displays model predictions for qualitative analysis.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install tensorflow numpy pandas matplotlib imageio
```

## Usage

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository>
```

### 2. Run Jupyter Notebook

Execute the notebook to train and evaluate the model:

```bash
jupyter notebook Image_segmentation_Unet_v2.ipynb
```

### 3. Utility Functions

#### Data Processing

The `utils.py` file includes functions for:

- **Loading images and masks**
- **Preprocessing images (resizing, normalization, etc.)**



#### Model Definition

U-Net model implementation is available in `utils.py`:

```python
from utils import unet_model
model = unet_model()
```

#### Display Results

```python
from utils import display

display([input_image, true_mask, predicted_mask])
```

## Results

The notebook provides visualizations of segmented outputs, including comparisons between ground truth masks and predictions.

## Future Work

- Improve model performance using data augmentation.
- Experiment with different loss functions.
- Extend to multi-class segmentation tasks.

## License

This project is licensed under the MIT License.

