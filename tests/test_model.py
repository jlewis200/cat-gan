#!/usr/bin/python3


import unittest
import torch
from torch.utils.data import DataLoader

from configs import get_model

DEVICE = "cpu"

class test_model(unittest.TestCase):
    """
    Model tests.
    """

    def test_model_functional(self):
        """
        Functional test to ensure the trained generator produces samples rated higher than a random image.
        """

        torch.manual_seed(0)
        model = get_model("cat-faces").to(DEVICE)
        checkpoint = torch.load("checkpoints/cat_faces_pretrained.pt", map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        n_samples = 32
        samples = model.sample(n_samples=n_samples)
        sample_score = model.forward(samples=samples)
        random_score = model.forward(samples=torch.rand_like(samples))
        self.assertTrue(torch.all(sample_score > random_score))