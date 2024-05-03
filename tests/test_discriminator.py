#!/usr/bin/python3


import unittest
import torch
from torch.utils.data import DataLoader

from configs import get_model

DEVICE = "cpu"

class test_discriminator(unittest.TestCase):
    """
    Discriminator tests.
    """

    def test_discriminator_integration(self):
        """
        Integration test to ensure the discriminator accepts input and produces output with appropriate type/shape.
        """
        n_samples = 32
        input_shape = torch.Size((n_samples, 3, 64, 64))
        expected_score_shape = torch.Size((n_samples,))
        model = get_model("cat-faces").to(DEVICE)
        fake_batch = torch.rand(input_shape) # uniform sample from [0, 1)
        fake_score = model.forward(samples=fake_batch)
        self.assertEqual(fake_score.shape, expected_score_shape)

    def test_discriminator_functional(self):
        """
        Functional test to ensure the trained discriminator produces a larger score for real images than random samples.
        """

        torch.manual_seed(0)
        model = get_model("cat-faces").to(DEVICE)
        checkpoint = torch.load("checkpoints/cat_faces_pretrained.pt", map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.dataset = model.get_dataset()
        dataloader = DataLoader(
            model.dataset,
            batch_size=32,
            shuffle=True
        )

        real_batch = next(iter(dataloader))[0]
        real_score = model.forward(samples=real_batch)
        fake_batch = torch.rand(real_batch.shape) # uniform sample from [0, 1)
        fake_score = model.forward(samples=fake_batch)
        self.assertTrue(torch.all(real_score > fake_score))