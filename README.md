# CamelyonPatch Classification with CNN and Vision Transformer

This repository contains a Jupyter notebook that performs binary classification on the CamelyonPatch dataset using two approaches: a Convolutional Neural Network (CNN) and a Vision Transformer (ViT). The project also includes explainability techniques using saliency maps and Grad-CAM to provide insights into the models' decisions.

## Overview

The CamelyonPatch dataset is a popular benchmark in the medical imaging domain, specifically for cancer detection in histopathological slides. The goal is to classify whether a given image patch contains tumor tissue.

## Features

1. **Binary Classification Models**:
   - **Convolutional Neural Network (CNN)**: A deep learning model designed for image data.
   - **Vision Transformer (ViT)**: A transformer-based architecture for image classification tasks.

2. **Explainability**:
   - **Saliency Maps**: Visualizations that highlight which parts of the input image influence the model's predictions.
   - **Grad-CAM**: Gradient-weighted Class Activation Mapping, used to generate heatmaps over input images to localize important regions for classification.

## File Contents

- `CamelyonPatch_Classification.ipynb`: The main notebook containing the entire workflow for training and evaluating the models, along with generating explainability visualizations.

## Dependencies

Ensure the following Python libraries are installed before running the notebook:
- TensorFlow or PyTorch (depending on the framework used for the models)
- NumPy
- Matplotlib
- OpenCV
- scikit-learn
- Transformers (for Vision Transformer)

You can install these dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/IsmailHatim/CameylonPatch-Classification
   cd CameylonPatch-Classification
   ```
2. Open the notebook:
   ```bash
   jupyter notebook CamelyonPatch_Classification.ipynb
   ```
3. Follow the instructions in the notebook to:
   - Train the CNN and Vision Transformer models.
   - Evaluate the models on test data.
   - Visualize explainability results using Saliency Maps and Grad-CAM.

## Results

The notebook demonstrates:
- The performance of CNN and Vision Transformer on the CamelyonPatch dataset.
- Explainability outputs to analyze model behavior:
  - Saliency maps show pixel-level importance.
  - Grad-CAM heatmaps highlight regions contributing to the predictions.