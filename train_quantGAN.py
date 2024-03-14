import os, sys
import pandas as pd
import numpy as np
import wandb
import torch
import torch.optim as optim
from tqdm import tqdm
from data_loader import *
from preprocess.acf import *
from preprocess.gaussianize import *
from QuantGAN.model import *
from utils import *

import joblib
import yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


### Hyperparameter ###
with open('./QuantGAN/configs/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
# manual_seed_all(42)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"  
device = torch.device(dev)


### Data ###
file_name = 'SP500_daily'
file_path = f'./QuantGAN/data/{file_name}.csv'
generator_path = './QuantGAN/result/'
data = pd.read_csv(file_path, parse_dates={'datetime': ['Date']}, date_format='%Y-%m-%d')

### Preprocess the data ###
df = data['Close']
log_returns = np.log(df/df.shift(1))[1:].to_numpy().reshape(-1, 1)
print(log_returns.shape)
standardScaler1 = StandardScaler()
standardScaler2 = StandardScaler()
gaussianize = Gaussianize()
log_returns_preprocessed = standardScaler2.fit_transform(gaussianize.fit_transform(standardScaler1.fit_transform(log_returns)))
print(log_returns_preprocessed.shape)

### Hyperparameter ###
with open('./QuantGAN/configs/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

lr = config['learning_rate']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
seq_len = config['seq_len']
clip = config['clip']
nz = config['z_dim']
receptive_field_size = config['receptive_field_size']

wandb.init(
   project='Quant-GAN',
   config = {
      "learning_rate": lr,
      "epochs": num_epochs,
      "batch_size": batch_size
    }
)

### Load the data ###
dataset = Loader32(log_returns_preprocessed, seq_len)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#### Model initialization ###
generator = Generator().to(device)
discriminator = Discriminator(seq_len).to(device)
gen_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)

### Train the model ###
t = tqdm(range(num_epochs))
for epoch in t:
    for idx, data in enumerate(dataloader, 0):
        
        discriminator.zero_grad()
        real = data.to(device)
        batch_size, seq_len = real.size(0), real.size(2)        
        noise = torch.randn(batch_size, nz, seq_len, device=device)
        fake = generator(noise).detach()
        disc_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
        disc_loss.backward()
        disc_optimizer.step()
        
        for dp in discriminator.parameters():
            dp.data.clamp_(-clip, clip)

        if idx % 5 == 0:
            generator.zero_grad()
            gen_loss = -torch.mean(discriminator(generator(noise)))
            gen_loss.backward()
            gen_optimizer.step()

    t.set_description('Discriminator Loss: %.8f Generator Loss: %.8f' % (disc_loss.item(), gen_loss.item()))
    
    wandb.log({"Generator_loss": gen_loss.item()})        
    wandb.log({"Discriminator_loss": disc_loss.item()})    

### Save the model ###
torch.save(generator, f'{generator_path}trained_generator_{file_name}_epoch_{epoch}.pth')

joblib.dump(standardScaler1, f'./QuantGAN/data/pickle/{file_name}_standardScaler1.pkl')
joblib.dump(standardScaler2, f'./QuantGAN/data/pickle/{file_name}_standardScaler2.pkl')
joblib.dump(gaussianize, f'./QuantGAN/data/pickle/{file_name}_gaussianize.pkl')
joblib.dump(log_returns, f'./QuantGAN/data/pickle/{file_name}_log_returns.pkl')
joblib.dump(log_returns_preprocessed, f'./QuantGAN/data/pickle/{file_name}_log_returns_preprocessed.pkl')

