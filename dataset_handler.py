# dataset_handler.py
import os
import random
from torchvision import transforms
from PIL import Image
import torch


class DatasetHandler:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
    def load_dataset(self):
        dataset_images = {}  # Ahora es un diccionario
        for folder in ["Defective", "Non-defective"]:
            folder_path = os.path.join(self.dataset_dir, folder)
            if os.path.isdir(folder_path):  # Verifica que sea un directorio válido
                dataset_images[folder] = [
                    os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".bmp")
                ]
        return dataset_images  # Devuelve un diccionario

    def load_dataset(self):
        dataset_paths = []
        for folder in ["Defective", "Non-defective"]:
            folder_path = os.path.join(self.dataset_dir, folder)
            for filename in os.listdir(folder_path):
                if filename.endswith(".bmp"):
                    dataset_paths.append(os.path.join(folder_path, filename))
        return dataset_paths

