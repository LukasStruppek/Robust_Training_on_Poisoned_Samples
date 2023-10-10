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


class TrainingConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    def create_model(self):
        model_config = self._config['model']
        print(model_config)
        model = Classifier(**model_config)
        return model

    def create_datasets(self):
        dataset_config = self._config['dataset']
        backdoor_config = self._config['backdoor']
        root_train = dataset_config['root_train'].lower()
        root_test = dataset_config['root_test'].lower()
        train_set, valid_set = None, None
        test_set_clean, test_set_poisoned = None, None

        data_transformation_train = self.create_transformations(
            mode='training')
        backdoor_transform = self.create_backdoor_transformations()
        train_set = BackdoorDataset(
            root=root_train,
            target_idx=self._config['backdoor']['target_idx'],
            transform=data_transformation_train,
            poison_first=backdoor_config['poison_first'],
            poisoning_rate=backdoor_config['poisoning_rate'],
            random_poisoning=backdoor_config['random_poisoning'],
            image_size=self.image_size,
            seed=self.seed,
            backdoor_transform=backdoor_transform,
            mode='train',
            remove_target_idx_samples=False)

        data_transformation_test = self.create_transformations(mode='test')
        test_set_clean = ImageFolder(root=root_test,
                                     transform=data_transformation_test)
        test_set_poisoned = BackdoorDataset(
            root=root_test,
            target_idx=self._config['backdoor']['target_idx'],
            transform=data_transformation_test,
            poison_first=False,
            poisoning_rate=1.0,
            random_poisoning=False,
            image_size=self.image_size,
            seed=self.seed,
            backdoor_transform=backdoor_transform,
            mode='test',
            remove_target_idx_samples=True)

        # Compute dataset lengths
        train_len, valid_len, test_len, test_poisoned = len(train_set), 0, 0, 0
        if valid_set:
            valid_len = len(valid_set)
        if test_set_clean:
            test_len = len(test_set_clean)
        if test_set_poisoned:
            test_poisoned = len(test_set_poisoned)

        print(
            f'Created {dataset_config["name"]} datasets with {train_len:,} training, {valid_len:,} validation, {test_len:,} clean test samples and {test_poisoned:,} poisoned test samples.\n',
            f'Transformations during training: {train_set.transform}\n',
            f'Transformations during evaluation: {test_set_clean.transform}')
        return train_set, valid_set, test_set_clean, test_set_poisoned

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

    def create_optimizer(self, model):
        optimizer_config = self._config['optimizer']
        for optimizer_type, args in optimizer_config.items():
            if not hasattr(optim, optimizer_type):
                raise ValueError(
                    f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
                )

            optimizer_class = getattr(optim, optimizer_type)
            optimizer = optimizer_class(model.parameters(), **args)
            break
        return optimizer

    def create_lr_scheduler(self, optimizer):
        if not 'lr_scheduler' in self._config:
            return None

        scheduler_config = self._config['lr_scheduler']
        for scheduler_type, args in scheduler_config.items():
            if not hasattr(optim.lr_scheduler, scheduler_type):
                raise ValueError(
                    f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class'
                )

            scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
            scheduler = scheduler_class(optimizer, **args)
        return scheduler

    def create_rtpt(self):
        rtpt_config = self._config['rtpt']
        rtpt = RTPT(name_initials=rtpt_config['name_initials'],
                    experiment_name=rtpt_config['experiment_name'],
                    max_iterations=self.training['num_epochs'])
        return rtpt

    @property
    def experiment_name(self):
        return self._config['experiment_name']

    @property
    def model(self):
        return self._config['model']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def optimizer(self):
        return self._config['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['lr_scheduler']

    @property
    def training(self):
        return self._config['training']

    @property
    def rtpt(self):
        return self._config['rtpt']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def wandb(self):
        return self._config['wandb']

    @property
    def image_size(self):
        return self._config['dataset']['image_size']

    @property
    def backdoor(self):
        return self._config['backdoor']

    @property
    def kd_temperature(self):
        return self._config['distillation']['temperature']

    @property
    def kd_alpha(self):
        return self._config['distillation']['alpha']

    @property
    def teacher_model(self):
        return self._config['distillation']['teacher_model']