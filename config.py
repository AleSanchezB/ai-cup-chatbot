# config.py
import torch

# Parámetros de configuración
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "path_to_pretrained_model"  # Si necesitas cargar un modelo preentrenado específico.
DATASET_DIR = "dataset_augmented"
IMAGE_DIM = (224, 224)  # Dimensiones de la imagen
MEAN = [0.485, 0.456, 0.406]  # Media para normalización
STD = [0.229, 0.224, 0.225]  # Desviación estándar para normalización
SIMILARITY_THRESHOLD = 0.6  # Umbral de similitud para la predicción
# SIMILARITY_THRESHOLD = 2.5
