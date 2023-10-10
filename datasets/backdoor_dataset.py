import math
import random

import numpy as np
import torch
from torchvision.datasets import ImageFolder

class BackdoorDataset(ImageFolder):

    def __init__(self,
                 root,
                 target_idx: int,
                 transform=None,
                 backdoor_transform=None,
                 poison_first: bool = True,
                 poisoning_rate: float = 0.1,
                 random_poisoning=False,
                 poison_target_class=False,
                 image_size=224,
                 seed=42,
                 mode='train',
                 remove_target_idx_samples=False):
        """
        Args:
            root (str): root directory of the dataset
            target_idx (int): index of the target class
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version. E.g, ``transforms.RandomCrop``
            backdoor_transform (callable, optional): A function/transform that takes in an image and returns a poisoned version. E.g, ``BlendedBackdoor``
            poison_first (bool): whether to poison the image before applying transformations
            poisoning_rate (float): percentage of the dataset to be poisoned
            random_poisoning (bool): whether to poison random samples during training
            poison_target_class (bool): whether to poison samples from the target class or not
            image_size (int): size of the image (image_size x image_size)
            seed (int): seed for index selection
        """
        super().__init__(root=root)
        self.root = root
        self.target_idx = target_idx
        self.transform = transform
        self.backdoor_transform = backdoor_transform
        self.poison_first = poison_first
        self.poisoning_rate = poisoning_rate
        self.random_poisoning = random_poisoning
        self.poison_target_class = poison_target_class
        self.image_size = image_size
        self.seed = seed
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.mode = mode

        # remove samples from the target class for evaluation
        if remove_target_idx_samples:
            sample_mask = np.array(self.targets) != self.target_idx
            self.samples = np.array(self.samples)[sample_mask]
            self.targets = np.array(self.targets)[sample_mask]

        # always add trigger to the same samples
        if self.random_poisoning is False:
            if self.poison_target_class:
                indices = range(len(self.targets))
            else:
                indices = np.argwhere(
                    np.array(self.targets) != target_idx).flatten()

            num_poisoned_samples = math.ceil(poisoning_rate * len(indices))
            np.random.seed(seed)
            self.poison_indices = np.random.choice(indices,
                                                   size=num_poisoned_samples,
                                                   replace=False)
        else:
            self.poison_indices = None

    def __getitem__(self, index: int):
        """
        Args:    
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = sample.resize((self.image_size, self.image_size))
        sample = np.array(sample)

        # check if current sample will be poisoned
        poison_sample = False
        # poison sample if target class is the ground-truth class
        if not self.poison_target_class and target == self.target_idx:
            poison_sample = False
        # poison sample if index is in poison_indices
        if self.random_poisoning is False and index in self.poison_indices:
            poison_sample = True
        # poison randomly
        elif self.random_poisoning and random.random() <= self.poisoning_rate:
            poison_sample = True

        # poison image before applying transformations
        if self.poison_first and poison_sample:
            if self.backdoor_transform:
                sample = self.backdoor_transform(sample)
                target = self.target_idx

        # apply standard transformations
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = int(target)

        # poison image after applying transformations
        if not self.poison_first and poison_sample:
            if self.backdoor_transform:
                sample = self.backdoor_transform(sample)
                target = self.target_idx

        if type(sample) == torch.Tensor:
            sample = sample.float()
        return sample, target

    def __len__(self):
        """
        Returns:
            int: number of samples in the dataset
        """
        return len(self.samples)
