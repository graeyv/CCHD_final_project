# Final Project - Cloud Computing for High-Dimensional Data (Fall Semester 2024, NTU)

This is our GitHub repository for the final project in the course **Cloud Computing for High-Dimensional Data** at NTU. The project is divided into two main parts:

1. **U-Net**: Training, pruning, and evaluating a 2D U-Net for binary segmentation using PyTorch.
2. **SAURON**: Implementation and evaluation of a novel pruning method for convolutional neural networks.

---

## Dataset
The dataset used for the U-Net part is from the [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge). It consists of 5,088 RGB images and their corresponding binary masks.

---

## Repository Structure

### U-Net
This folder contains all files related to the training, pruning, and evaluation of a 2D U-Net model:

1. **`main.ipynb`**  
   - The main Jupyter Notebook that orchestrates the training, pruning, and evaluation workflow.
   - Includes step-by-step instructions and visualizations of model performance and metrics.

2. **`custom_functions.py`**  
   - Contains utility functions for various pruning techniques:
     - Structured and unstructured pruning (local and global).
     - Functions for evaluating model performance, including accuracy, IoU, and Dice coefficient.
     - Methods to count weights and filters in the model.

3. **`custom_model.py`**  
   - Defines the architecture of the custom U-Net model using PyTorch's `nn.Module`.
   - Encoder-decoder structure with skip connections and transposed convolutions for upsampling.

4. **`dataset.py`**  
   - Implements the `CarDataset` class to handle the dataset.
   - Supports image and mask transformations for pre-processing.
   - Converts binary masks to tensors for PyTorch compatibility.

---

## Usage Instructions

1. **Download and Prepare Data**  
   - Download the dataset from the [Carvana Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge).
   - Organize the dataset locally:
     ```
     data/train       # Folder containing training images
     data/train_masks # Folder containing corresponding binary masks
     ```

2. **Adapt Paths in Code**  
   - Update the paths in `dataset.py` or any other relevant files to match your local directory structure.

---

## Authors
- Pierre Lacoste
- Romain Englebert
- Yves Grädel
