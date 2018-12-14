import math
import torch
import torch.nn as nn
from torch.nn.functional import linear


# Naive implementation: noise sampled for __every__ forward step
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = 0.5

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def scale_noise(size):
        # noinspection PyUnresolvedReferences
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        # ugly way to separate training and eval
        if self.training:
            in_noise = self.scale_noise(self.in_features)
            out_noise = self.scale_noise(self.out_features)
        else:
            in_noise = torch.zeros(self.in_features, dtype=torch.float32)
            out_noise = torch.zeros(self.out_features, dtype=torch.float32)
        epsilon_w = out_noise.ger(in_noise)
        epsilon_b = out_noise

        w = self.weight_mu + self.weight_sigma * epsilon_w
        b = self.bias_mu + self.bias_sigma * epsilon_b
        return linear(x, w, b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias
        )
