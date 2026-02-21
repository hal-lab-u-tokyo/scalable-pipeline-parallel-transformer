import os
from torchvision import datasets, transforms


def _imagenet(root: str):
    transforms_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transforms_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds_train = datasets.ImageFolder(
        os.path.join(root, "ImageNet-1K", "train"), transforms_train
    )
    ds_val = datasets.ImageFolder(
        os.path.join(root, "ImageNet-1K", "val"), transforms_val
    )
    return ds_train, ds_val


def _cifar10(root: str):
    transforms_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )
    transforms_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    root = os.path.join(root, "cifar-10")
    ds_train = datasets.CIFAR10(
        root, train=True, transform=transforms_train, download=True
    )
    ds_val = datasets.CIFAR10(
        root, train=False, transform=transforms_val, download=True
    )
    return ds_train, ds_val


def _cifar100(root: str):
    transforms_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]),
        ]
    )
    transforms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]),
        ]
    )
    root = os.path.join(root, "cifar-100")
    ds_train = datasets.CIFAR100(
        root, train=True, transform=transforms_train, download=True
    )
    ds_val = datasets.CIFAR100(
        root, train=False, transform=transforms_val, download=True
    )
    return ds_train, ds_val


def _cinic10(root: str):
    transforms_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )
    transforms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    cinic_root = os.path.join(root, "CINIC-10")
    ds_train = datasets.ImageFolder(os.path.join(cinic_root, "train"), transforms_train)
    ds_val = datasets.ImageFolder(os.path.join(cinic_root, "valid"), transforms_val)
    # ds_val = datasets.ImageFolder(os.path.join(cinic_root, "test"), transforms_val)
    return ds_train, ds_val
