import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np

# WGAN-GP 참조
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

def manual_seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False
    random.seed(seed)

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1, 1), device="cuda:0")
    alpha = alpha.expand(real_samples.size())

    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    d_interpolates = discriminator(interpolates)
    fake = torch.autograd.Variable(torch.ones(d_interpolates.shape), requires_grad=False)
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



