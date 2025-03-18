import albumentations as A
import cv2
import os
import numpy as np

class DataAugmenter:
    def __init__(self, input_dir, output_dir, num_aug=5):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_aug = num_aug

        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)

        # Definir las transformaciones de data augmentation
        self.transform = A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.Transpose(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.HueSaturationValue(),
        ])

    def augment_and_save_images(self):
        """Aplica aumentación a todas las imágenes del input_dir y las guarda en output_dir."""
        for img_name in os.listdir(self.input_dir):
            img_path = os.path.join(self.input_dir, img_name)

            # Cargar imagen con OpenCV
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error cargando {img_name}, saltando...")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Aplicar transformaciones y guardar imágenes aumentadas
            for i in range(self.num_aug):
                augmented = self.transform(image=image)['image']
                save_path = os.path.join(self.output_dir, f"aug_{i}_{img_name}")
                cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        print(f"Aumentación completada. Imágenes guardadas en {self.output_dir}")
