# image_processor.py
from PIL import Image
import torchvision.transforms as transforms

class ImageProcessor:
    def __init__(self, resize_dim=(224, 224), mean=None, std=None):
        self.transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean or [0.485, 0.456, 0.406], std=std or [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path, device):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(device)
