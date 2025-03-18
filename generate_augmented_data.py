from data_augmenter import DataAugmenter

def main():
    # Directorios de entrada y salida
    dataset_dirs = ["dataset/Defective", "dataset/Non-defective"]
    output_dirs = ["dataset_augmented/Defective", "dataset_augmented/Non-defective"]

    for input_dir, output_dir in zip(dataset_dirs, output_dirs):
        augmenter = DataAugmenter(input_dir, output_dir, num_aug=5)
        augmenter.augment_and_save_images()

if __name__ == "__main__":
    main()
