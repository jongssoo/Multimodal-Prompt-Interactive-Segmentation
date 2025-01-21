# Multimodal-Prompt-Interactive-Segmentation

## Model Training

This repository is for training a model consisting of three components: **Mask_Segmentor**, **Text_Predictor**, and **Sequence Module**. Below is the guide to properly set up the dataset, preprocess the data, and run the training process.

## Dataset Structure

To train the model, the dataset needs to be structured as follows:
```bash
Dataset
├── Firefly
│ ├── train
│ └── val
├── HRF
├── CHASE_DB1
└── ...
```
Each dataset folder (e.g., `Firefly`, `HRF`) should contain subdirectories for `train` and `val` sets.

## Preprocessing

The preprocessing of the dataset is handled in `dataset.py`. This script processes the raw dataset and prepares it for training. Make sure that the dataset is correctly structured before running the preprocessing steps.

## Training the Model

To train the model, use the `train.py` script. The script takes care of the training process for each model component:

1. **Mask_Segmentor**
2. **Text_Predictor**
3. **Sequence Module**

### Training Workflow

1. **Dataset Setup**: Ensure that the dataset is in the correct folder structure as shown above.
2. **Preprocessing**: Run the preprocessing steps defined in `dataset.py`.
3. **Model Training**: The actual training and validation processes are managed in `function.py`.
4. **Model Configuration**: Model settings and dataset paths are configured in `cfg.py`.

After setting up all the configurations, you can start the training by running:

```bash
python train.py
