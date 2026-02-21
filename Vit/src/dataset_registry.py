from dataclasses import dataclass
from typing import Callable, Tuple
from torch.utils.data import Dataset, Subset
from .dataset_factory import _cifar10, _cifar100, _cinic10, _imagenet


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int
    shape: Tuple[int, int, int]
    builder_fn: Callable[[str], tuple[Dataset, Dataset]]


REGISTRY: dict[str, DatasetSpec] = {
    "cifar-10": DatasetSpec(10, (3, 32, 32), _cifar10),
    "cifar-100": DatasetSpec(100, (3, 32, 32), _cifar100),
    "cinic-10": DatasetSpec(10, (3, 32, 32), _cinic10),
    "ImageNet-1K": DatasetSpec(1000, (3, 224, 224), _imagenet),
}


def load_dataset(name: str, root: str, subset_N: int | None = None):
    spec = REGISTRY[name]
    ds_tr, ds_va = spec.builder_fn(root)
    if subset_N:
        ds_tr = Subset(ds_tr, range(min(len(ds_tr), subset_N)))
        ds_va = Subset(ds_va, range(min(len(ds_va), subset_N)))
    return ds_tr, ds_va
