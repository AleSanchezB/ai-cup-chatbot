# similarity_comparer.py
import torch

class SimilarityComparer:
    def __init__(self, model, image_processor, threshold=0.5):
        self.model = model
        self.image_processor = image_processor
        self.threshold = threshold
    
    def compare_with_dataset(self, image_path, dataset_paths, device):
        input_image = self.image_processor.preprocess_image(image_path, device)
        similarities = []

        with torch.no_grad():
            for ref_image_path in dataset_paths:
                ref_image = self.image_processor.preprocess_image(ref_image_path, device)
                similarity = self.model(input_image, ref_image).item()
                similarities.append((ref_image_path, similarity))
        
        similarities.sort(key=lambda x: x[1])  # Ordenar por similitud (menor distancia es más parecido)
        return similarities
    
    def infer_class(self, image_path, dataset_paths, device):
        similarities = self.compare_with_dataset(image_path, dataset_paths, device)
        
        # Contadores para las clases
        defective_count = 0
        non_defective_count = 0

        # Iterar sobre las similitudes y contar
        for path, sim in similarities:
            print(f"sim: {sim} in path: {path}")
            if sim >= self.threshold:
                if 'Defective' in path:
                    defective_count += 1
                elif 'Non-defective' in path:
                    non_defective_count += 1

    
        # Inferir la clase basada en el conteo
        if defective_count > non_defective_count:
            return 'Defective'
        else:
            return 'Non-defective'
