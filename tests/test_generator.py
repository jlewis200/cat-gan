#!/usr/bin/python3


import unittest
import torch
from torch.utils.data import DataLoader

from configs import get_model

DEVICE = "cpu"

class test_generator(unittest.TestCase):
    """
    Generator tests.
    """

    def test_generator_integration(self):
        """
        Integration test to ensure the generator produces produces output with appropriate type/shape/bounds.
        """

        n_samples = 32
        expected_sample_shape = torch.Size((n_samples, 3, 64, 64))
        model = get_model("cat-faces").to(DEVICE)
        samples = model.sample(n_samples=n_samples)
        self.assertEqual(samples.shape, expected_sample_shape)
        self.assertTrue(torch.all(samples >= 0))
        self.assertTrue(torch.all(samples <= 1))