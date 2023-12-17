import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import v2

from domain.utils.DataAugmentation import DataAugmentation
import warnings


class DatasetLoader:
    def __init__(
        self,
        batch_size=4,
        data_augmentation_flag=False,
        image_size=32,
        grayscale=True,
    ):
        self._batch_size = batch_size
        warnings.filterwarnings("ignore")

        preprocessing_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if grayscale:
            preprocessing_list.append(transforms.Grayscale(num_output_channels=3))
            
        preprocessing_list.extend(
            [
                transforms.ToTensor(),
            ]
        )
        
        if not grayscale:
            preprocessing_list.extend(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    # transforms.RandomErasing(p=0.5),
                ]
            )

        self._preprocessing = transforms.Compose(preprocessing_list)

        self._transform = self._preprocessing

        if data_augmentation_flag:
            data_augmentation = DataAugmentation()
            self._transform = transforms.Compose(
                [data_augmentation.get_transforms(), self._preprocessing]
            )

        self._classes = [str(i) for i in range(10)]

    """
    load MNIST dataset
    if dataset does not exist, download it
    clip_dataset_count: clip the train/test dataset to the first n data
    """

    def load_mnist_dataset(self, clip_dataset_count=None):
        self._train_dataset = torchvision.datasets.MNIST(
            root="./dataset/MNIST",
            train=True,
            download=True,
            transform=self._transform,
        )

        if clip_dataset_count is not None:
            self._train_dataset = torch.utils.data.Subset(
                self._train_dataset, range(clip_dataset_count)
            )

        self._train_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=4,
        )

        self._test_dataset = torchvision.datasets.MNIST(
            root="./dataset/MNIST",
            train=False,
            download=True,
            transform=self._preprocessing,
        )

        if clip_dataset_count is not None:
            self._test_dataset = torch.utils.data.Subset(
                self._test_dataset, range(clip_dataset_count)
            )

        self._test_loader = torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=4,
        )

    def load_cat_dog_dataset(self, clip_dataset_count=None):
        self._train_dataset = torchvision.datasets.ImageFolder(
            root="./dataset/Dataset_Cvdl_Hw2_Q5/dataset/training_dataset",
            transform=self._transform,
        )

        if clip_dataset_count is not None:
            self._train_dataset = torch.utils.data.Subset(
                self._train_dataset, range(clip_dataset_count)
            )

        self._train_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=4,
        )

        self._test_dataset = torchvision.datasets.ImageFolder(
            root="./dataset/Dataset_Cvdl_Hw2_Q5/dataset/validation_dataset",
            transform=self._preprocessing,
        )

        if clip_dataset_count is not None:
            self._test_dataset = torch.utils.data.Subset(
                self._test_dataset, range(clip_dataset_count)
            )

        self._test_loader = torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=4,
        )

    """
    getters
    """

    def get_data_loader(self):
        return self._train_loader, self._test_loader

    def get_classes(self):
        return self._classes

    def get_batch_size(self):
        return self._batch_size

    # only do normalization
    def get_preprocessing(self):
        return self._preprocessing

    # only do data augmentation (without normalization)
    def get_transform(self):
        return self._transform
