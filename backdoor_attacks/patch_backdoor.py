import torch
import numpy as np
import torchvision.transforms as T
import random
from PIL import ImageColor


class PatchBackdoor(torch.nn.Module):

    def __init__(self,
                 trigger_type='checkerboard',
                 trigger_size=3,
                 trigger_loc='bottom_right'):
        """
        Args:
            trigger_type (str): type of the trigger. Currently only 'checkerboard', 'white', and 'black' are supported, as well as RBG values in the hex format.
            trigger_size (int): size of the trigger (trigger_size x trigger_size)
            trigger_loc (str): location of the trigger. Currently only 'bottom_right' and 'random' are supported.
        """
        super().__init__()
        self.trigger_type = trigger_type.strip().lower()
        self.trigger_size = trigger_size
        self.trigger = self.create_trigger()
        self.trigger_loc = trigger_loc.strip().lower()

    def create_trigger(self):
        """
        Creates the trigger
        """
        if self.trigger_type == 'checkerboard':
            trigger = torch.tensor(
                np.indices(
                    (self.trigger_size, self.trigger_size, 1)).sum(axis=0) % 2)
        elif 'checkerboard_' in self.trigger_type:
            color_1 = ImageColor.getcolor(
                self.trigger_type.split('_')[1], "RGB")
            color_2 = ImageColor.getcolor(
                self.trigger_type.split('_')[2], "RGB")
            trigger = torch.zeros((self.trigger_size, self.trigger_size, 3))
            for ch in range(3):
                trigger[:, :, ch] = color_1[ch] / 255.0
                trigger[::2, ::2, ch] = color_2[ch] / 255.0
                trigger[1::2, 1::2, ch] = color_2[ch] / 255.0

        elif self.trigger_type == 'white':
            trigger = torch.ones((self.trigger_size, self.trigger_size, 1))
        elif self.trigger_type == 'black':
            trigger = torch.zeros((self.trigger_size, self.trigger_size, 1))
        elif self.trigger_type[0] == '#' and len(self.trigger_type) == 7:
            patch_color = ImageColor.getcolor(self.trigger_type, "RGB")
            trigger = torch.zeros((self.trigger_size, self.trigger_size, 3))
            trigger[:, :, 0].fill_(patch_color[0] / 255.0)
            trigger[:, :, 1].fill_(patch_color[1] / 255.0)
            trigger[:, :, 2].fill_(patch_color[2] / 255.0)
        return trigger

    def forward(self, img):
        """
        Applies the trigger to the image.
        Args:
            img (torch.Tensor or np.ndarray): image to be poisoned
        Returns:    
            img (torch.Tensor or np.ndarray): poisoned image
        """
        if isinstance(img, np.ndarray):
            if self.trigger_loc == 'bottom_right':
                img[-self.trigger_size:,
                    -self.trigger_size:, :] = self.trigger * 255
            elif self.trigger_loc == 'random':
                x = random.Random(img.sum()).randint(
                    0, img.shape[0] - self.trigger_size)
                y = random.Random(img.sum()).randint(
                    0, img.shape[1] - self.trigger_size)
                img[x:x + self.trigger_size,
                    y:y + self.trigger_size] = self.trigger * 255
            else:
                raise ValueError(
                    f'Location {self.trigger_loc} is not supported.')
        elif isinstance(img, torch.Tensor):
            if self.trigger_loc == 'bottom_right':
                img[:, -self.trigger_size:,
                    -self.trigger_size:] = self.trigger.permute(2, 0, 1)
            elif self.trigger_loc == 'random':
                x = random.Random(img.sum()).randint(
                    0, img.shape[1] - self.trigger_size)
                y = random.Random(img.sum()).randint(
                    0, img.shape[2] - self.trigger_size)
                img[:, x:x + self.trigger_size,
                    y:y + self.trigger_size] = self.trigger.permute(2, 0, 1)
            else:
                raise ValueError(
                    f'Location {self.trigger_loc} is not supported.')
        else:
            raise TypeError(
                f'Datatype {type(img)} is not compatible. Please convert the input to numpy.ndarray or torch.Tensor first.'
            )
        return img

    def __repr__(self):
        """
        String representation of the class
        """
        return f"{self.__class__.__name__}()"
