#!/usr/bin/python3

"""
This module is used to initialize the model for the various datasets.
"""

# pylint: disable=no-member
import numpy as np

from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import (
    ToTensor,
    Compose,
    Resize,
    Normalize,
    ColorJitter,
    RandomHorizontalFlip
)

from model import GAN


def get_model(config):
    """
    Initialize the model according to the supplied config string.
    """

    model = None

    if config == "cat-faces":
        model = get_cat_faces()

    return model


def get_cat_faces():
    """
    Get model configured for the celeba dataset.
    """

    model = GAN()

    dataset_kwargs = {"root": "~/datasets/cat_faces", "transform": ToTensor()}

    # use a lambda to set the config but delay loading until training
    model.get_dataset = lambda: ImageFolder(**dataset_kwargs)

    # mean/std used by GAN to scale model input
    mean = 0.5
    std = 0.5
    #model.transform_in = Normalize(mean, std)
    model.transform_in = Normalize(mean, std)

    # mean/std used by GAN to scale targets for the loss function
    # map [0, 1] to [-1, 1]
    model.transform_target = Normalize(0.5, 0.5)

    # map reconstruction range [-1, 1] to PIL range [0, 1]
    model.transform_out = Normalize(-1, 2)

    return model
