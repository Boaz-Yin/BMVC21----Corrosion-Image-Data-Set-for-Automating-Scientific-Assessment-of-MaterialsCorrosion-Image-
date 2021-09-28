from typing import Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision
import yacs.config


class CenterCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.CenterCrop(
            config.dataset.image_size)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class Normalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image


class RandomCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomApply(torch.nn.ModuleList([torchvision.transforms.RandomCrop(
            config.tta.resize,
            padding=config.augmentation.random_crop.padding,
            fill=config.augmentation.random_crop.fill,
            padding_mode=config.augmentation.random_crop.padding_mode)]), config.augmentation.random_crop.prob)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)

class GaussianBlur:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomApply(torch.nn.ModuleList([torchvision.transforms.GaussianBlur(
            config.augmentation.gaussianblur.kernel_size,
            config.augmentation.gaussianblur.sigma)]), config.augmentation.gaussianblur.prob)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)

class ColorJitter:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomApply(torch.nn.ModuleList([torchvision.transforms.ColorJitter(
            config.augmentation.colorjitter.brightness,
            config.augmentation.colorjitter.contrast,
            config.augmentation.colorjitter.saturation,
            config.augmentation.colorjitter.hue)]), config.augmentation.colorjitter.prob)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)

class RandomResizeCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomApply(torch.nn.ModuleList([torchvision.transforms.RandomResizedCrop(
            config.tta.resize, config.augmentation.random_resize_crop.scale, config.augmentation.random_resize_crop.aspect_ratio)]), config.augmentation.random_resize_crop.prob)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class RandomHorizontalFlip:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomHorizontalFlip(
            config.augmentation.random_horizontal_flip.prob)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)

class RandomVerticalFlip:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomVerticalFlip(
            config.augmentation.random_vertical_flip.prob)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)

class RandomPerspective:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomPerspective(
            config.augmentation.random_perspective.distortion_scale, config.augmentation.random_perspective.prob)
    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)

class RandomRotation:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomApply(torch.nn.ModuleList([torchvision.transforms.RandomRotation(
            config.augmentation.random_rotation.degrees, expand=config.augmentation.random_rotation.expand)]),config.augmentation.random_rotation.prob)
    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        self.transform(data) 

class Resize:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.Resize(config.tta.resize)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class ToTensor:
    def __call__(
        self, data: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(data, tuple):
            return tuple([self._to_tensor(image) for image in data])
        else:
            return self._to_tensor(data)

    @staticmethod
    def _to_tensor(data: np.ndarray) -> torch.Tensor:
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))
