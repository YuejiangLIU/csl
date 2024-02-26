# Adapted from DomainBed

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "DomainNet",
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class DomainLabelDataset(Dataset):
    """A wrapper for ImageFolder to include domain labels."""
    def __init__(self, dataset, domain_label):
        self.dataset = dataset
        self.domain_label = domain_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, self.domain_label

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, envs, split):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.datasets = []
        for i, environment in enumerate(environments):
            if i in envs:
                path = os.path.join(root, environment, split)
                env_dataset = ImageFolder(path, transform)
                domain_dataset = DomainLabelDataset(env_dataset, domain_label=i)
                self.datasets.append(domain_dataset)
                print("Loaded data from ", path)
 
        self.dataset = ConcatDataset(self.datasets)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].dataset.classes)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, envs, split):
        self.dir = os.path.join(root, "domainnet/")
        super().__init__(self.dir, envs, split)
