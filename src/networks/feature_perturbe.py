import torch
import torch.nn as nn
import random


class FeaturePerturbation(nn.Module):
    def __init__(self, lam=0.9, kap=0.2, eps=1e-6, use_gpu=True):
        super(FeaturePerturbation, self).__init__()
        # self.num_features = num_features
        self.eps = eps
        self.lam = lam
        self.kap = kap
        self.use_gpu = use_gpu

    def forward(self, x):
        # normalization
        mu = x.mean(dim=[2, 3], keepdim=True)  # [B,C,1,1]
        var = x.var(dim=[2, 3], keepdim=True)  # [B,C,1,1]
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        batch_mu = mu.mean(dim=[0], keepdim=True)  # [1,C,1,1]
        batch_psi = (mu.var(dim=[0], keepdim=True) +
                     self.eps).sqrt()  # [1,C,1,1]
        batch_sig = sig.mean(dim=[0], keepdim=True)  # [1,C,1,1]
        batch_phi = (sig.var(dim=[0], keepdim=True) +
                     self.eps).sqrt()  # [1,C,1,1]
        epsilon = torch.empty(1).uniform_(-self.kap, self.kap)
        gamma = self.lam * sig + (1 - self.lam) * \
            batch_sig + epsilon * batch_phi
        beta = self.lam * mu + (1 - self.lam) * batch_mu + epsilon * batch_psi
        x_aug = gamma * x_normed + beta
        return x_aug


if __name__ == '__main__':
    module = FeaturePerturbation().cuda()
    inputs = torch.randn(4, 128, 256, 256)
    outpust = module(inputs)
    print(outpust.size())
