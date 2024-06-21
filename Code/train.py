import os, sys
import joblib, yaml, wandb
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from data_loader import *
from preprocess.gaussianize import *
from QuantGAN_stock.Code.model import *
from utils import *

path = '/workspace/QuantGAN_stock'
try:
    os.chdir(path)
    print("Current working directory: {0}".format(os.getcwd()))
except FileNotFoundError:
    print("Directory {0} does not exist".format(path))
    
### Set seed
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

### Hyperparameter ###
with open('./configs/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
file_name = config['file_name']
lr = config['lr']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
seq_len = config['seq_len']
clip = config['clip']
nz = config['z_dim']
receptive_field_size = config['receptive_field_size']
num_G = config['num_G']

### Wandb initialization
wandb.init(
   project='Quant-GAN',
   config = {
      "learning_rate": lr,
      "batch_size": batch_size,
      "clip": clip
    }
)

### Make the directory for saving the results    
full_name = f'num_G_{num_G}_{num_epochs}_{batch_size}_lr_{lr}_{wandb.run.name}_clip_{clip}'        
if not os.path.isdir(f'./Results/{full_name}/'):
    os.mkdir(f'./Results/{full_name}/')

### Read the Dataset  
with open('./Dataset/' + file_name +'.csv', 'rb') as fh:
    df = pd.read_csv(fh)
    
### Data Preprocessing   
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)
df = df.apply(pd.to_numeric).astype(float)
log_returns = np.diff(np.log(df), axis=0)
log_returns_preprocessed = scaling(log_returns, num_G)  

### Load the data ###
dataset = dataloader(log_returns_preprocessed, seq_len)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#### Model initialization ###
generator = Generator().to(device)
discriminator = Discriminator(seq_len).to(device)
criterion = nn.BCELoss()

gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

### Train the model ###
t = tqdm(range(num_epochs))
for epoch in t:
    for idx, data in enumerate(dataloader, 0):
        
        real = data.to(device)
        batch_size, seq_len = real.size(0), real.size(2)        
        noise = torch.randn(batch_size, nz, seq_len, device=device)
        
        # Train the discriminator
        discriminator.zero_grad()
        fake = generator(noise).detach()
        
        # Real and fake labels
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)
        
        # Discriminator loss for real and fake data
        real_output = discriminator(real).view(-1)
        fake_output = discriminator(fake).view(-1)
        
        disc_loss_real = criterion(real_output, real_labels)
        disc_loss_fake = criterion(fake_output, fake_labels)
        
        disc_loss = disc_loss_real + disc_loss_fake
        disc_loss.backward()
        disc_optimizer.step()        
        
        generator.zero_grad()
        fake = generator(noise)
        output = discriminator(fake).view(-1)
        gen_loss = criterion(output, real_labels)  # We want the generator to fool the discriminator
        gen_loss.backward()
        gen_optimizer.step()

    t.set_description('Discriminator Loss: %.8f Generator Loss: %.8f' % (disc_loss.item(), gen_loss.item()))
    
    wandb.log({"Generator_loss": gen_loss.item()})        
    wandb.log({"Discriminator_loss": disc_loss.item()})    
    
    ### 모델 결과값 저장 - 생성자의 State_dict
    if epoch % 2 == 0:
        for i in range(num_G):            
            torch.save(generator.state_dict(), f'./Results/{full_name}/Generator_{epoch}.pt')
            print(f"Save the Generator_{epoch}")                              


