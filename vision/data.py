import torch
import torchvision
from torch.utils.data import Subset, ConcatDataset
from math import ceil
import tqdm

import warnings
warnings.filterwarnings("ignore")

from domainbed import DomainNet

import pdb

RESIZE, CROP = 256, 224
TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(RESIZE),
        torchvision.transforms.CenterCrop(CROP),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_imagenet(datapath, split, batch_size, shuffle, transform=TRANSFORM):
    ds = torchvision.datasets.ImageNet(root=datapath, split=split, transform=transform)
    loader = torch.utils.data.DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=min(batch_size//16, 8)) # <-- add num_workers=min(batch_size//16, 8)
    return ds, loader


def split_imbal_dataset(dataset, n_imbal=4, p_imbal=0.0, num_classes=1000):
    images_per_class = len(dataset) // num_classes
    subset_size = len(dataset) // (n_imbal+1)

    count_per_class_imbal = images_per_class * n_imbal // (n_imbal+1)
    count_per_class_last = images_per_class // (n_imbal+1)
    subsets = []

    imbal_indices = [[] for _ in range(n_imbal)]
    for i in range(n_imbal):
        class_start = i * num_classes // n_imbal
        class_end = (i + 1) * num_classes // n_imbal

        for j in range(num_classes):
            if class_start <= j < class_end:
                start_idx = j * images_per_class
                imbal_indices[i].extend(range(start_idx, start_idx + count_per_class_imbal))

    for i in range(n_imbal):
        indices = imbal_indices[i]
        # Ensure the subset size is exactly 1/5th of the dataset
        assert len(indices) == subset_size
        subsets.append(Subset(dataset, indices))

    # Last subset with equal representation
    equal_indices = []
    for j in range(num_classes):
        start_idx = j * images_per_class + count_per_class_imbal
        equal_indices.extend(range(start_idx, start_idx + count_per_class_last))

    subsets.append(Subset(dataset, equal_indices))

    return subsets


def get_imbal_loader(datapath, split, n_train, batch_size, n_imbal, transform=TRANSFORM):
    dataset = torchvision.datasets.ImageNet(root=datapath, split=split, transform=transform)
    subsets = split_imbal_dataset(dataset, n_imbal)

    combined_indices = []
    for i in range(n_imbal):
        combined_indices.extend(subsets[i].indices)
    original_dataset = subsets[0].dataset
    supset = Subset(original_dataset, combined_indices)
    loaders_tr = [torch.utils.data.DataLoader(supset, shuffle=True, batch_size=batch_size, num_workers=min(batch_size//16, 8))]

    loader_te = torch.utils.data.DataLoader(subsets[-1], shuffle=False, batch_size=batch_size, num_workers=min(batch_size//16, 8))
    return loaders_tr, loader_te


def get_imbal_sampler(dataset, start_class, end_class, ratio=0.0, num_classes=1000):
    # Assign weights to each class
    class_weights = [1.0 if start_class <= i < end_class else ratio for i in range(num_classes)]

    # Assign weights to each sample
    sample_weights = [class_weights[dataset.targets[i]] for i in range(len(dataset))]

    # Create a WeightedRandomSampler
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    return sampler


def get_domain_lables(loader):
    all_z = []

    for x, y, z in tqdm.tqdm(loader):
        all_z.append(z)

    all_z = torch.cat(all_z, axis=0)
    return all_z


def get_domainnet(datapath, split, batch_size, shuffle, envs=[0, 1, 2, 3, 4, 5]):
    ds = DomainNet(datapath, envs, split)
    loader = torch.utils.data.DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=min(batch_size//16, 8)) # <-- add num_workers=min(batch_size//16, 8)
    return ds, loader