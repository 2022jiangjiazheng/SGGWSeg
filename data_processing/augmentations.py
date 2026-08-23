# Project repository: https://github.com/2022jiangjiazheng
"""Image and mask augmentations used by the Great Wall dataset loader."""

import numpy as np
import torch
import tormentor
import torchvision


class MyWrap(object):
    """Random elastic wrap augmentation provided by Tormentor."""

    def __call__(self, image, zone):
        wrap_rand = tormentor.Wrap.override_distributions(
            roughness=tormentor.random.Uniform(value_range=(.1, .7)),
            intensity=tormentor.random.Uniform(value_range=(.0, .2)),
        )
        wrap = wrap_rand()
        image = wrap(image)
        zone = wrap(zone, is_mask=True)
        return image, zone


class Rotate(object):
    """Rotate an image and its mask by 90, 180, or 270 degrees."""

    def __call__(self, image, zone):
        random = np.random.randint(0, 3)
        angle = 90
        if random == 1:
            angle = 180
        elif random == 2:
            angle = 270
        image = torchvision.transforms.functional.rotate(image, angle=angle)
        zone = torchvision.transforms.functional.rotate(zone, angle=angle)
        return image, zone


class Bright(object):
    """Apply a random brightness adjustment while preserving no-data pixels."""

    def __call__(self, image, zone):
        bright_rand = tormentor.Brightness.override_distributions(
            brightness=tormentor.random.Uniform((-0.2, 0.2))
        )
        bright = bright_rand()
        image_transformed = bright(image.clone())
        image_transformed[image == 0] = 0.0
        return image_transformed, zone


class Noise(object):
    """Apply multiplicative Gaussian noise and clamp the image to [0, 1]."""

    def __call__(self, image, zone):
        noise = torch.normal(mean=0, std=0.1, size=image.shape)
        image = image + image * noise
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0
        return image, zone
