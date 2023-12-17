import torch
from PIL import Image
from torchvision import transforms


class DataAugmentation:
    def __init__(self):
        self.transforms = transforms.Compose(
            [
                # transforms.RandomResizedCrop(size=(224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(
                    brightness=(0.7, 1.3),
                    contrast=(0.7, 1.3),
                    saturation=(0.7, 1.3),
                    hue=(-0.1, 0.1),
                ),
            ]
        )

    def load_image_from_path_with_pil(self, path: str) -> Image:
        return Image.open(path)

    """
    do data augmentation on the image
    """

    def transform(self, img: Image) -> Image:
        return self.transforms(img)

    """
    get the transforms of data augmentation
    """

    def get_transforms(self):
        return self.transforms