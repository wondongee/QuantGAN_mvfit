import torch
import torch.optim as optim
from model.loss import * 
import torch.nn.functional as F

def train(idx,
          data,
          discriminator,
          generator,
          disc_optimizer,
          gen_optimizer,
          config,
          device):

    loss_func = config['model']['loss_func']
    clip = config['model']['clip']
    nz = config['model']['z_dim']
    
    if loss_func not in ["WGAN", "WGAN-GP", "SN-GAN", "VANILLA"]:
        raise ValueError(f"Invalid loss function: {loss_func}")
        
    discriminator.zero_grad()
    real = data.to(device)
    batch_size, seq_len = real.size(0), real.size(2)
    
    noise = torch.randn(batch_size, nz, seq_len, device=device)
    # ! VERY IMPORTANT: detach() is needed to avoid backpropagating through the generator
    fake = generator(noise).detach().to(device)
    

    if loss_func =="WGAN-GP":
        lambda_gp = 10
        disc_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
        gradient_penalty = compute_gradient_penalty(discriminator, real.data, fake.data, device)
        disc_loss += lambda_gp * gradient_penalty
        
    # TODO
    elif loss_func == "WGAN":
        disc_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
        
    else:
        disc_loss_real = F.binary_cross_entropy_with_logits(discriminator(real), torch.ones_like(discriminator(real)))
        disc_loss_fake = F.binary_cross_entropy_with_logits(discriminator(fake), torch.zeros_like(discriminator(fake)))
        disc_loss = disc_loss_real + disc_loss_fake
    
    disc_loss.backward()
    disc_optimizer.step()
    
    if loss_func == "WGAN": 
        for dp in discriminator.parameters():
            dp.data.clamp_(-clip, clip)

    if idx % config['train']['n_critic'] == 0:
        generator.zero_grad()
        # todo: Check regenerating noise
        # noise = torch.randn(batch_size, nz, seq_len, device=device)  #? For mode collapse
        # ?
        gen_loss = -torch.mean(discriminator(generator(noise)))
        #gen_loss = -torch.mean(torch.log(discriminator(generator(noise))))
        gen_loss.backward()
        gen_optimizer.step()
    
    else:
        gen_loss = None 
        
    return disc_loss, gen_loss, discriminator, generator, disc_optimizer, gen_optimizer