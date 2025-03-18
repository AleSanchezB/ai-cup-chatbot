# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from torchvision import models
# from PIL import Image
# import numpy as np

# # Definimos la arquitectura de la red siamesa
# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         base_model = models.resnet18(pretrained=True)
#         self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Quitamos la última capa
#         self.fc = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
    
#     def forward(self, img1, img2):
#         feat1 = self.feature_extractor(img1)
#         feat2 = self.feature_extractor(img2)
#         feat1 = feat1.view(feat1.size(0), -1)
#         feat2 = feat2.view(feat2.size(0), -1)
#         distance = torch.abs(feat1 - feat2)
#         output = self.fc(distance)
#         output = torch.sigmoid(output)  # Aplicamos Sigmoid para restringir la salida en [0, 1]
#         return output

# # Preprocesamiento de imágenes
# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     image = Image.open(image_path).convert("RGB")
#     return transform(image).unsqueeze(0).to(device)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Cargar el modelo
# model = SiameseNetwork().to(device)
# model.eval()

# # Función para comparar una imagen con el dataset
# def compare_with_dataset(image_path, dataset_paths, model):
#     input_image = preprocess_image(image_path)
#     similarities = []
    
#     with torch.no_grad():
#         for ref_image_path in dataset_paths:
#             ref_image = preprocess_image(ref_image_path)
#             similarity = model(input_image, ref_image).item()
#             similarities.append((ref_image_path, similarity))
    
#     similarities.sort(key=lambda x: x[1])  # Ordenar por similitud (menor distancia es más parecido)
#     return similarities

# import os

# # Ruta a la carpeta donde están las imágenes
# dataset_dir = "dataset"

# # Cargar imágenes de ambas carpetas
# def load_dataset(dataset_dir):
#     dataset_paths = []
    
#     # Obtener las rutas de todas las imágenes en las carpetas Defective y Non-defective
#     for folder in ["Defective", "Non-defective"]:
#         folder_path = os.path.join(dataset_dir, folder)
#         for filename in os.listdir(folder_path):
#             if filename.endswith(".bmp"):  # Asegúrate de que solo se carguen archivos .bmp
#                 dataset_paths.append(os.path.join(folder_path, filename))
    
#     return dataset_paths


# def infer_class(image_path, dataset_paths, model):
#     # Obtener los resultados de similitud
#     similarities = compare_with_dataset(image_path, dataset_paths, model)
    
#     # Contadores para las clases
#     defective_count = 0
#     non_defective_count = 0

#     # Umbral de similitud para considerarlo una coincidencia
#     threshold = 0.5

#     # Iterar sobre las similitudes y contar
#     for path, sim in similarities:
#         if sim >= threshold:
#             print(sim)
#             if 'Defective' in path:
#                 defective_count += 1
#             elif 'Non-defective' in path:
#                 non_defective_count += 1

#     # Inferir la clase basada en el conteo
#     if defective_count > non_defective_count:
#         return 'Defective'
#     else:
#         return 'Non-defective'

# # Ejemplo de uso
# if __name__ == "__main__":
#     test_image = "test/Non-Defective/testnondefective.bmp"  # Ruta a la imagen a comparar
#     dataset_images = load_dataset(dataset_dir)
    
#     # Inferir la clase de la imagen de prueba
#     predicted_class = infer_class(test_image, dataset_images, model)
    
#     # Mostrar la clase inferida
#     print(f"La imagen de prueba pertenece a la clase: {predicted_class}")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Definimos la arquitectura de la Triplet Network
class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(2048, 256)  # Embedding de 256 dimensiones

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        feat = F.normalize(feat, p=2, dim=1)  # Normalización L2
        return feat

class WeightedTripletLoss(nn.Module):
    def __init__(self, margin=1.0, weight=2.0):
        super(WeightedTripletLoss, self).__init__()
        self.margin = margin
        self.weight = weight
        self.loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=margin)

    def forward(self, anchor, positive, negative, label):
        loss = self.loss_fn(anchor, positive, negative)
        return loss * (self.weight if label == "Non-defective" else 1.0)

# Definimos la función de pérdida con distancia euclidiana
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

# Preprocesamiento de imágenes
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo
model = TripletNetwork().to(device)
model.eval()

# Cargar imágenes del dataset
def load_dataset(dataset_dir):
    dataset_paths = []
    for folder in ["Defective", "Non-defective"]:
        folder_path = os.path.join(dataset_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".bmp"):
                dataset_paths.append(os.path.join(folder_path, filename))
    return dataset_paths

# Comparar imagen con dataset
def compare_with_dataset(image_path, dataset_paths, model):
    input_image = preprocess_image(image_path)
    similarities = []

    with torch.no_grad():
        for ref_image_path in dataset_paths:
            ref_image = preprocess_image(ref_image_path)
            anchor_feat = model(input_image)
            ref_feat = model(ref_image)
            distance = F.pairwise_distance(anchor_feat, ref_feat).item()
            similarities.append((ref_image_path, distance))
    
    similarities.sort(key=lambda x: x[1])  # Menor distancia es más parecido
    return similarities

def infer_class(image_path, dataset_paths, model):
    similarities = compare_with_dataset(image_path, dataset_paths, model)
    defective_count, non_defective_count = 0, 0
    threshold = 0.5  # Ajustar según evaluación

    for path, dist in similarities:
        if dist < threshold:
            if 'Defective' in path:
                defective_count += 1
            elif 'Non-defective' in path:
                non_defective_count += 1
    
    return 'Defective' if defective_count > non_defective_count else 'Non-defective'

# Evaluación con AUC-ROC
def evaluate_model(model, test_loader):
    model.eval()
    all_labels, all_scores = [], []
    with torch.no_grad():
        for img_anchor, img_positive, img_negative in test_loader:
            anchor_feat = model(img_anchor.to(device))
            positive_feat = model(img_positive.to(device))
            negative_feat = model(img_negative.to(device))
            pos_dist = F.pairwise_distance(anchor_feat, positive_feat)
            neg_dist = F.pairwise_distance(anchor_feat, negative_feat)
            score = pos_dist - neg_dist
            all_scores.extend(score.cpu().numpy())
            all_labels.extend([1] * len(pos_dist) + [0] * len(neg_dist))
    auc = roc_auc_score(all_labels, all_scores)
    print(f"AUC-ROC Score: {auc}")

# Optimización
def train_model(model, train_loader, num_epochs=20):
    criterion = WeightedTripletLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for img_anchor, img_positive, img_negative in train_loader:
            img_anchor, img_positive, img_negative = img_anchor.to(device), img_positive.to(device), img_negative.to(device)
            optimizer.zero_grad()
            anchor_feat, positive_feat, negative_feat = model(img_anchor), model(img_positive), model(img_negative)
            loss = criterion(anchor_feat, positive_feat, negative_feat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step(total_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

def visualize_embeddings(model, dataset_images):
    embeddings = extract_embeddings(model, dataset_images)
    
    # Perplexity debe ser < n_samples
    n_samples = len(embeddings)
    perplexity = min(30, n_samples - 1)  # Evita que sea mayor a n_samples

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plot_embeddings(embeddings_2d, dataset_images)

def plot_embeddings(embeddings_2d, dataset_images):
    plt.figure(figsize=(8, 6))
    
    # Extraer etiquetas desde la ruta de las imágenes
    labels = [path.split("/")[-2] for path in dataset_images]  
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    label_color_map = {label: color for label, color in zip(unique_labels, colors)}

    for i, label in enumerate(labels):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=label_color_map[label], label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.legend()
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.title("Visualización de Embeddings con t-SNE")
    plt.show()
    
def extract_embeddings(model, dataset_images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for image_path in dataset_images:
            image = preprocess_image(image_path)  # Asegúrate de que preprocess_image esté bien definido
            embedding = model.feature_extractor(image).view(image.size(0), -1)
            embeddings.append(embedding.cpu().numpy())

    return np.vstack(embeddings) 
    
# Ejemplo de uso
if __name__ == "__main__":
    dataset_dir = "dataset"
    dataset_images = load_dataset(dataset_dir)
    
    visualize_embeddings(model, dataset_images)

    test_image = "test/Non-Defective/testnondefective.bmp"
    predicted_class = infer_class(test_image, dataset_images, model)
    print(f"La imagen de prueba pertenece a la clase: {predicted_class}")

    test_image = "test/Defective/testdefective.bmp"
    predicted_class = infer_class(test_image, dataset_images, model)
    print(f"La imagen de prueba pertenece a la clase: {predicted_class}")
