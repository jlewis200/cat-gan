#!/usr/bin/python3

"""
Generative Adversarial Network (GAN) model.
"""

# ignore these pylint findings due to torch lazy loading false positives
# pylint: disable=no-member, no-name-in-module

from time import time

import torch

from torch import nn, tanh
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.nn.functional import one_hot
from torch.optim import RMSprop


class GAN(nn.Module):
    """
    Generative Adversarial Network (GAN) model.
    """

    def __init__(self, bits=8):
        super().__init__()

        self.generator = Generator(bits=bits)
        self.discriminator = Discriminator()

        # register the training epoch and start time so it is saved with the model's state_dict
        self.register_buffer("epoch", torch.tensor(0, dtype=int))

        # the model is identified by the initial creation time
        self.register_buffer("start_time", torch.tensor(int(time()), dtype=int))

        # track the total number of generator training steps
        self.register_buffer("gen_steps", torch.tensor(0, dtype=int))

    def clip_params(self, param_abs=0.01):
        """
        Clip the discriminator parameters.
        """

        with torch.no_grad():
            for param in self.discriminator.parameters():
                param.clip_(-param_abs, param_abs)

    def forward(self, n_samples=32, samples=None):
        """
        Perform the forward training pass.  If samples are provided, classify real/fake.
        If samples are not provided, generate n_samples and classify real/fake.
        """

        if samples is None:
            samples = self.sample(n_samples)

        return self.discriminator(self.transform_in(samples))

    def sample(self, n_samples, temp=1.0):
        """
        Sample from the model.
        """

        # sample from the generator and apply the output transformation
        return self.transform_out(self.generator(n_samples, temp=temp))

    @property
    def device(self):
        """
        Return the pytorch device where input tensors should be located.
        """

        return self.discriminator.device


class Discriminator(nn.Module):
    """
    GAN Discriminator class.
    Adapted from https://github.com/Zeleni9/pytorch-wgan/blob/master/models/dcgan.py
    """

    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential()

        self.blocks.append(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1
            )
        )
        self.blocks.append(nn.BatchNorm2d(num_features=128))
        self.blocks.append(nn.LeakyReLU(0.2))

        self.blocks.append(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
            )
        )
        self.blocks.append(nn.BatchNorm2d(num_features=256))
        self.blocks.append(nn.LeakyReLU(0.2))

        self.blocks.append(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
            )
        )
        self.blocks.append(nn.BatchNorm2d(num_features=512))
        self.blocks.append(nn.LeakyReLU(0.2))

        self.blocks.append(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1
            )
        )
        self.blocks.append(nn.BatchNorm2d(num_features=1024))
        self.blocks.append(nn.LeakyReLU(0.2))

        self.blocks.append(
            nn.Conv2d(
                in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0
            )
        )

    @property
    def device(self):
        """
        Return the pytorch device where input tensors should be located.
        """

        return self.blocks[0].weight.device

    def forward(self, tensor):
        """
        Perform a forward pass through the discriminator.
        """

        return self.blocks(tensor).flatten(start_dim=0)


class Generator(nn.Module):
    """
    GAN Generator class.
    Adapted from https://github.com/Zeleni9/pytorch-wgan/blob/master/models/dcgan.py
    """

    def __init__(self, n_mixtures=10, bits=8):
        super().__init__()

        self.blocks = nn.Sequential()

        self.blocks.append(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0
            )
        )
        self.blocks.append(nn.BatchNorm2d(num_features=512))
        self.blocks.append(nn.ReLU(True))

        self.blocks.append(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            )
        )
        self.blocks.append(nn.BatchNorm2d(num_features=256))
        self.blocks.append(nn.ReLU(True))

        self.blocks.append(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
            )
        )
        self.blocks.append(nn.BatchNorm2d(num_features=128))
        self.blocks.append(nn.ReLU(True))

        self.blocks.append(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
            )
        )
        self.blocks.append(nn.BatchNorm2d(num_features=64))
        self.blocks.append(nn.ReLU(True))

        self.blocks.append(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            )
        )

        self.dmol_net = DmolNet(64, n_mixtures, bits)

    def forward(self, n_samples, temp=1):
        """
        Perform the forward pass through this group.
        """

        tensor = temp * torch.randn((n_samples, 512, 1, 1), device=self.device)

        return self.dmol_net(self.blocks(tensor))

    @property
    def device(self):
        """
        Get the pytorch device where input tensors should be located.
        """

        return self.blocks[0].weight.device


class GanOptimizer:
    """
    Combine the generator/discriminator optimizers together in a convenient package.
    """

    def __init__(self, model, learning_rate):
        """
        Store the generator/discriminator optimizers.
        """

        self.gen = RMSprop(
            model.generator.parameters(), lr=learning_rate, maximize=True
        )
        self.dis = RMSprop(
            model.discriminator.parameters(), lr=learning_rate, maximize=True
        )
        self.dis_scaler = None

        if model.device.type == "cuda":
            self.dis_scaler = torch.cuda.amp.GradScaler()

    def zero_grad(self):
        """
        Zero parameters of each optimizer.
        """

        self.gen.zero_grad()
        self.dis.zero_grad()

    def dis_backward(self, loss):
        """
        Perform the backward pass.  Scale if scaler available.
        """

        if self.dis_scaler is not None:
            self.dis_scaler.scale(loss).backward()
            self.dis_scaler.unscale_(self.dis)

        else:
            loss.backward()

    def gen_step(self):
        """
        Take a generator optimization step.
        """

        self.gen.step()

    def dis_step(self):
        """
        Take a discriminator optimization step.
        """

        if self.dis_scaler is not None:
            self.dis_scaler.step(self.dis)
            self.dis_scaler.update()

        else:
            self.dis.step()

    def set_lr(self, learning_rate):
        """
        Override the learning rate the optimizer was initialized with.
        """

        for param_group in self.gen.param_groups + self.dis.param_groups:
            param_group["lr"] = learning_rate


class DmolNet(nn.Module):
    """
    Discrete Mixture Of Logistics network.
    """

    def __init__(self, channels, n_mixtures, bits=8):
        super().__init__()

        self.n_mixtures = n_mixtures
        self.bits = bits

        # convs for generating red sub-pixel distribution parameters
        self.r_mean = nn.Conv2d(channels, n_mixtures, 1)
        self.r_logscale = nn.Conv2d(channels, n_mixtures, 1)

        # convs for generating green sub-pixel distribution parameters
        self.g_mean = nn.Conv2d(channels, n_mixtures, 1)
        self.g_logscale = nn.Conv2d(channels, n_mixtures, 1)
        self.gr_coeff = nn.Conv2d(channels, n_mixtures, 1)

        # convs for generating blue sub-pixel distribution parameters
        self.b_mean = nn.Conv2d(channels, n_mixtures, 1)
        self.b_logscale = nn.Conv2d(channels, n_mixtures, 1)
        self.br_coeff = nn.Conv2d(channels, n_mixtures, 1)
        self.bg_coeff = nn.Conv2d(channels, n_mixtures, 1)

        # conv for generating the log-probabilities of each of the n mixtures
        self.logits = nn.Conv2d(channels, n_mixtures, 1)

    def forward(self, dec_out):
        """
        Sample from the discrete logistic distribution.
        """

        # get the distibutions/distribution-log-probabilities from the generator output
        dlog_r, dlog_g, dlog_b, logits = self.get_distributions(dec_out)

        # get the color channel reparameterized samples from all n_mixtures
        # reparameterize to enable backprop through the sampling
        color_r = dlog_r.rsample()
        color_g = dlog_g.rsample()
        color_b = dlog_b.rsample()
        # shape N x n_mixtures x H x W

        # randomly choose 1 of n_mixtures distributions per-pixel, based on their log probabilities
        # torch Categorical treats the last dim as the category
        # permute the n_mixtures dimension to the final dimension and sample
        indexes = Categorical(logits=logits.permute(0, 2, 3, 1)).sample()
        # shape N x H x W, value: an int in [0, n_mixtures - 1]

        # one hot encode the final dimension and permute to mixture dimension
        indexes = one_hot(indexes, num_classes=self.n_mixtures).permute(0, 3, 1, 2)
        # shape N x n_mixtures x H x W

        # indexes now has a value of 1 in the channel corresponding with the selected distribution
        # all others are zero
        # pointwise multiply with the color samples and sum along the channels axis
        # to zeroize all channels except those of the selected distributions
        color_r = (color_r * indexes).sum(dim=1, keepdim=True)
        color_g = (color_g * indexes).sum(dim=1, keepdim=True)
        color_b = (color_b * indexes).sum(dim=1, keepdim=True)
        # color_* shape N x 1 x H x W

        # stack the color channels
        # clamping to the valid range is consistent with attributing the probability of -1/1 the
        # remaining probability density toward -inf/inf
        img = torch.cat((color_r, color_g, color_b), dim=1).clamp(-1, 1)
        # shape N x 3 x H x W

        return img

    def get_distributions(self, dec_out):
        """
        Get the distributions for training/sampleing.
        """

        # use a numerical stability term to prevent very small scales after exponentiation
        stability = -7

        # dist parameters for the red sub-pixel distributions
        r_mean = self.r_mean(dec_out)
        r_logscale = self.r_logscale(dec_out).clamp(min=stability)
        # shape N x n_mixtures x H x W

        # dist parameters for the blue sub-pixel distributions
        g_mean = self.g_mean(dec_out)
        g_logscale = self.g_logscale(dec_out).clamp(min=stability)
        # shape N x n_mixtures x H x W

        # green-red mixing coefficient
        gr_coeff = tanh(self.gr_coeff(dec_out))
        # shape N x n_mixtures x H x W

        # dist parameters for the green sub-pixel distributions
        b_mean = self.b_mean(dec_out)
        b_logscale = self.b_logscale(dec_out).clamp(min=stability)
        # shape N x n_mixtures x H x W

        # blue-red/blue-green mixing coefficient
        br_coeff = tanh(self.br_coeff(dec_out))
        bg_coeff = tanh(self.bg_coeff(dec_out))
        # shape N x n_mixtures x H x W

        # g_mean/b_mean are mixed with the distribution means for conditional sampling
        # mix the mean of green sub-pixel with red
        g_mean = g_mean + (gr_coeff * r_mean)
        # shape N x n_mixtures x H x W

        # mix the mean of blue sub-pixel with red/green
        b_mean = b_mean + (br_coeff * r_mean) + (bg_coeff * g_mean)
        # shape N x n_mixtures x H x W

        # initialize the distributions
        dlog_r = DiscreteLogistic(r_mean, torch.exp(r_logscale), self.bits)
        dlog_g = DiscreteLogistic(g_mean, torch.exp(g_logscale), self.bits)
        dlog_b = DiscreteLogistic(b_mean, torch.exp(b_logscale), self.bits)

        # log probability of each mixture for each distribution
        logits = self.logits(dec_out)
        # shape N x n_mixtures x H x W

        return dlog_r, dlog_g, dlog_b, logits


class DiscreteLogistic(TransformedDistribution):
    """
    Discretized Logistic distribution.  Models values in the range [-1, 1] discretized into
    2**n_bits + 1 buckets.  Probability density outside of [-1, 1] is attributed to the upper/lower
    value as appropriate.  Central values are given density:

    CDF(val + half bucket width) - CDF(val - half bucket width)

    If a value has extremely low probability, it is assigned PDF(val) for numerical stability.
    """

    def __init__(self, mean, scale, bits=8):
        # bits parameterizes the width of the discretization bucket
        # higher bits -> smaller buckets
        # half of the width of the bucket
        self.half_width = 1 / ((2 ** bits) - 1)

        # this continuous logistic snippet adapted from pytorch distribution docs
        base_distribution = Uniform(torch.zeros_like(mean), torch.ones_like(mean))
        transforms = [SigmoidTransform().inv, AffineTransform(loc=mean, scale=scale)]
        super().__init__(base_distribution, transforms)

    def log_prob(self, value):
        """
        The DiscreteLogistic distribution is built from a TransformedDistribution to create a
        continuous logistic distribution.  This log_prob function uses the TransformedDistribution's
        cdf() and log_prob() functions to return the discretized log probability of value.
        """

        # use a numerical stability term to prevent log(0)
        stability = 1e-12
        prob_threshold = 1e-5

        # use the non-discrete log-probability as a base
        # this value is used when prob < prob_threshold
        # this would indicate the distribution parameters are off by quite a bit
        log_prob = super().log_prob(value)

        # find the discrete non-log probability of laying within the bucket
        prob = self.cdf(value + self.half_width) - self.cdf(value - self.half_width)

        # if the non-log probability is above a threshold,
        # replace continuous log-probability with the discrete log-probability
        mask = prob > prob_threshold
        log_prob[mask] = prob[mask].clamp(min=stability).log()

        # edge case at -1:  replace -1 with cdf(-1 + half bucket width)
        mask = value <= -1
        log_prob[mask] = (
            self.cdf(value + self.half_width)[mask].clamp(min=stability).log()
        )

        # edge case at 1:  replace 1 with (1 - cdf(1 - half bucket width))
        mask = value >= 1
        log_prob[mask] = (
            (1 - self.cdf(value - self.half_width))[mask].clamp(min=stability).log()
        )

        return log_prob

    def entropy(self):
        """
        Not required for this use case.
        """

        raise NotImplementedError

    def enumerate_support(self, expand=True):
        """
        Not required for this use case.
        """

        raise NotImplementedError

    @property
    def mean(self):
        """
        Not required for this use case.
        """

        raise NotImplementedError

    @property
    def mode(self):
        """
        Not required for this use case.
        """

        raise NotImplementedError

    @property
    def variance(self):
        """
        Not required for this use case.
        """

        raise NotImplementedError
