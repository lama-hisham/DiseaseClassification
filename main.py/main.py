import torch
import os
from .data.loader import get_data_loader
from .data.split import split_data
from .model.architecture import define_model
from .training.trainer import train_model
from .evaluation.evaluator import test_model


# Set the dataset path and model parameters
dataset_path = r'C:\Users\Lenovo\Downloads\extractedcitrus\Citrus'
num_classes = 20  # number of plant disease classes

# Preprocess the dataset
transform = get_preprocessed_transform()
dataset = ImageFolder(dataset_path, transform=transform)

# Split the dataset into training, validation, and testing sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size