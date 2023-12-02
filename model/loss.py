import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# WGAN-GP 참조
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.size(0), 1, 1), device=device)
    alpha = alpha.expand(real_samples.size())
    #print("alpha size :", alpha.shape)

    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    d_interpolates = discriminator(interpolates)
    fake = torch.autograd.Variable(torch.ones(d_interpolates.shape), requires_grad=False).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def wasserstein_loss(real, fake):
    return - (torch.mean(real) - torch.mean(fake))


