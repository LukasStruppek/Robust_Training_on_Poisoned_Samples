from types import SimpleNamespace
from typing import List

import numpy as np
import torch.optim as optim
import torchvision.transforms as T
import yaml
from models.classifier import Classifier
from rtpt.rtpt import RTPT
from torchvision.datasets import *

from datasets.backdoor_dataset import BackdoorDataset
from torchvision.datasets import ImageFolder

from backdoor_attacks.blended_backdoor import BlendedBackdoor
from backdoor_attacks.patch_backdoor import PatchBackdoor


class VariationConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    def create_datasets(self):
        dataset_config = self._config['dataset']
        backdoor_config = self._config['backdoor']

        data_transformation = self.create_transformations(mode='test')
        backdoor_transform = self.create_backdoor_transformations()
        dataset = BackdoorDataset(
            root=dataset_config['root'],
            target_idx=backdoor_config['target_idx'],
            transform=data_transformation,
            poison_first=backdoor_config['poison_first'],
            poisoning_rate=backdoor_config['poisoning_rate'],
            random_poisoning=backdoor_config['random_poisoning'],
            poison_target_class=backdoor_config['poison_target_class'],
            image_size=self.image_size,
            seed=self.seed,
            backdoor_transform=backdoor_transform,
            mode='train',
            remove_target_idx_samples=False)

        print(
            f'Created {dataset_config["name"]} datasets with {len(dataset)} training samples.\n',
            f'Transformations during training: {data_transformation}')
        return dataset

    def create_backdoor_transformations(self):
        backdoor_config = self._config['backdoor']
        if 'BlendedBackdoor' in backdoor_config:
            backdoor_transform = BlendedBackdoor(
                **backdoor_config['BlendedBackdoor'])
        elif 'PatchBackdoor' in backdoor_config:
            backdoor_transform = PatchBackdoor(
                **backdoor_config['PatchBackdoor'])
        else:
            raise NotImplementedError(
                f'Backdoor type {backdoor_config} not implemented')
        return backdoor_transform

    def create_transformations(self, mode, to_tensor=True):
        """
        mode: 'training' or 'test'
        """
        dataset_config = self._config['dataset']
        transformation_list = []
        if to_tensor:
            transformation_list.append(T.ToTensor())
        if mode == 'training' and 'transformations' in self._config:
            transformations = self._config['transformations']
            if transformations != None:
                for transform, args in transformations.items():
                    if not hasattr(T, transform):
                        raise ValueError(
                            f'{transform} is no valid transformation. Please write the type exactly as the Torchvision class'
                        )
                    else:
                        transformation_class = getattr(T, transform)
                        transformation_list.append(
                            transformation_class(**args))

        elif mode == 'test':
            image_size = dataset_config['image_size']
            transformation_list.append(
                T.Resize((image_size, image_size), antialias=True))
        else:
            raise ValueError(f'{mode} is no valid mode for augmentation')

        data_transformation = T.Compose(transformation_list)

        return data_transformation

    def create_rtpt(self, max_iterations):
        rtpt_config = self._config['rtpt']
        rtpt = RTPT(name_initials=rtpt_config['name_initials'],
                    experiment_name=rtpt_config['experiment_name'],
                    max_iterations=max_iterations)
        return rtpt

    @property
    def rtpt(self):
        return self._config['rtpt']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def image_size(self):
        return self._config['dataset']['image_size']

    @property
    def backdoor(self):
        return self._config['backdoor']

    @property
    def output_folder(self):
        return self._config['output_folder']

    @property
    def diffusion(self):
        return self._config['diffusion']

    @property
    def color_transfer(self):
        return self._config['diffusion']['color_transfer']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def batch_size(self):
        return self._config['dataset']['batch_size']