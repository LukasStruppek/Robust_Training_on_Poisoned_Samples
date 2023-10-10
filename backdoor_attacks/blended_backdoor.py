import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision


class BlendedBackdoor(torch.nn.Module):

    def __init__(self, image_size, trigger_type='random', blend_ratio=0.1):
        """
        Args:
            image_size (int): size of the image (image_size x image_size)
            trigger_type (str): type of the trigger. Currently only 'random' and 'hello_kitty' are supported.
            blend_ratio (float): ratio of the trigger to be blended with the image
        """
        super().__init__()
        self.image_size = image_size
        self.trigger_type = trigger_type
        self.trigger = self.create_trigger()
        self.blend_ratio = blend_ratio

    def create_trigger(self):
        """
        Creates the trigger
        """
        if self.trigger_type == 'random':
            generator = torch.Generator().manual_seed(42)
            trigger = torch.randint(low=0,
                                    high=255,
                                    size=(self.image_size, self.image_size, 3),
                                    generator=generator) / 255.0
        elif self.trigger_type == 'hello_kitty':
            trigger = torchvision.io.read_image(
                'backdoor_attacks/imgs/hello_kitty.png')
            trigger = TF.resize(trigger, (self.image_size, self.image_size),
                                antialias=True)
            trigger = trigger.permute(1, 2, 0) / 255.0
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
            img = (1 - self.blend_ratio) * img + (
                self.blend_ratio) * self.trigger.numpy() * 255
            img = img.astype(np.uint8)
        elif isinstance(img, torch.Tensor):
            img = (1 - self.blend_ratio) * img + (
                self.blend_ratio) * self.trigger.permute(2, 0, 1)
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
