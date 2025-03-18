# main.py
import torch
from model import SiameseNetwork, TripletSiameseNetwork
from image_processor import ImageProcessor
from dataset_handler import DatasetHandler
from similarity_comparer import SimilarityComparer
import config

def main(test_image_path):
    # Inicializar los componentes
    model = SiameseNetwork().to(config.DEVICE)
    # model = TripletSiameseNetwork().to(config.DEVICE)
    model.eval()

    image_processor = ImageProcessor(resize_dim=config.IMAGE_DIM, mean=config.MEAN, std=config.STD)
    dataset_handler = DatasetHandler(config.DATASET_DIR)
    similarity_comparer = SimilarityComparer(model=model, image_processor=image_processor, threshold=config.SIMILARITY_THRESHOLD)

    # Cargar las imágenes del dataset
    dataset_images = dataset_handler.load_dataset()

    # Inferir la clase de la imagen de prueba
    predicted_class = similarity_comparer.infer_class(test_image_path, dataset_images, config.DEVICE)
    
    # Mostrar la clase inferida
    print(f"La imagen de prueba pertenece a la clase: {predicted_class}")


# Ejemplo de uso
if __name__ == "__main__":
    test_image_non_defective = "test/Non-Defective/testnondefective.bmp"  # Ruta a la imagen a comparar
    test_image_defective = "test/Defective/testdefective.bmp"  # Ruta a la imagen a comparar
    main(test_image_non_defective)
    main(test_image_defective)
