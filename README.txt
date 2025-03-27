# Oxford-IIIT Pet Image Segmentation

This repository contains the code for the Computer Vision mini-project focused on image segmentation using the Oxford-IIIT Pet Dataset.

## Project Description

This project explores and compares different deep learning models for segmenting pet images from the Oxford-IIIT Pet Dataset. The models implemented include U-Net, Autoencoder pre-training, and a CLIP-based model. Additionally, the project includes a prompt-based interactive segmentation tool.

## Repository Contents

The repository includes the following files:

* **augmentation.py:** Contains code for applying various data augmentation techniques, including geometric transformations, color-based adjustments, and noise-based perturbations.
   
* **checking.ipynb:** Jupyter Notebook for checking or debugging purposes.
   
* **Clip_model.keras:** Keras model file for the CLIP-based segmentation model.
   
* **data_loading.py:** Python script for loading and preprocessing the Oxford-IIIT Pet Dataset.
   
* **experiment.py:** Python script for running experiments.
   
* **main2_a.ipynb, main2_b.ipynb, main2_c.ipynb, main2_d.ipynb:** Jupyter Notebooks containing the main training and evaluation code for different models.
   
* **metrics.py:** Python script defining the evaluation metrics (e.g., IoU, Dice coefficient).
   
* **models.py:** Python script containing the implementations of the U-Net, Autoencoder, and CLIP segmentation models.
   
* **perturbations_plots.ipynb:** Jupyter Notebook for generating plots related to robustness analysis under various perturbations.
   
* **point_segmentation.py:** Python script for point-based segmentation.
   
* **test.ipynb:** Jupyter Notebook for testing the trained models.
   
* **ui.ipynb:** Jupyter Notebook for the User Interface, likely for the prompt-based segmentation tool.
   
* **\_\_pycache__/:** Python cache directory.

## Model Implementations

The project implements the following image segmentation models:

* **U-Net:** A U-shaped encoder-decoder architecture for image segmentation.
   
* **Autoencoder:** An unsupervised learning model used for pre-training an encoder for the segmentation task.
   
* **CLIP-based Model:** A segmentation model utilizing the pre-trained CLIP Vision Transformer (ViT) encoder.
   
* **Prompt-CLIP Model:** An adaptation of the CLIP model for interactive segmentation, allowing users to guide the segmentation using prompts.

## Data Augmentation

Data augmentation techniques were employed to improve model robustness and generalization. The following types of augmentations were used:

* Geometric transformations (e.g., flipping, rotation, translation).
   
* Color-based adjustments (e.g., brightness, contrast, saturation, hue).
   
* Noise and filtering (e.g., Gaussian noise, blurring, JPEG compression).

## Experimental Results

The CLIP-based model demonstrated superior performance compared to the other models. The robustness of the CLIP model was evaluated under various perturbations.

## Usage

For specific usage instructions, please refer to the individual Jupyter Notebooks and Python scripts.

## Dependencies

* TensorFlow
* Keras
* TensorFlow Datasets
* Transformers
* OpenCV
* Hugging Face
