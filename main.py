import os, sys
import pandas as pd
import numpy as np
import wandb
import torch
import torch.optim as optim
from tqdm import tqdm
from data_loader import *
from model import *
from utils import *
from train import *
from preprocess.preprocessor import *
import yaml
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_model(config, device):    
    discriminator = Discriminator(config['model']['seq_len']).to(device)
    generator = Generator().to(device)
    return generator, discriminator

def init_data_loader(data, config):
    dataset = Loader32(data, config['model']['seq_len'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)
    return dataloader

def init_optimizer(generator, discriminator, config):
    gen_optimizer = optim.Adam(generator.parameters(), config['train']['gen_learning_rate'], betas=(0.5, 0.99))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=config['train']['disc_learning_rate'], betas=(0.5, 0.99))
    return gen_optimizer, disc_optimizer

## ## ## 

def main(config_file):
        
    # Get Hyperparameters
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)         
    device = torch.device(config['train']['device']) if torch.cuda.is_available() else torch.device("cpu")
    
    # Initialize wandb
    wandb_config = {**config['train'], **config['model']}
    wandb.init(project='Quant-GAN', config=wandb_config)
    
    file_name = config['data']['file_name']
    file_path = os.path.join(config['data']['file_path'], config['data']['file_name'])
    generator_path = config['data']['generator_path']
    
    # Make the directory for saving the results
    full_name = os.path.join(generator_path, f'{file_name.split(".")[0]}_{wandb.run.name}')
    if not os.path.isdir(full_name):
        os.mkdir(full_name)    

    # Load data        
    try:
        data = pd.read_csv(file_path, parse_dates={'datetime': ['Date']}, date_format='%Y-%m-%d')
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        sys.exit()            
    
    # Process data
    log_returns_preprocessed = processor(data, config['data']['file_path'], config['data']['file_name'])
    logger.info("Data shape: %s", log_returns_preprocessed.shape)

    # init model loader, optimizer
    generator, discriminator = init_model(config, device)
    generator.train()
    discriminator.train()
    dataloader = init_data_loader(log_returns_preprocessed, config)
    gen_optimizer, disc_optimizer = init_optimizer(generator, discriminator, config)

    # Train the model
    last_gen_loss = None
    max_train_steps = config['train']['num_epochs'] * len(dataloader)
    progress_bar = tqdm(range(max_train_steps))
    
    for epoch in range(config['train']['num_epochs']):
        for idx, data in enumerate(dataloader):
        
            disc_loss, gen_loss, discriminator, generator, disc_optimizer, gen_optimizer = train(
                idx, data, discriminator, generator, disc_optimizer, gen_optimizer, 
                config, device
            )         
            if gen_loss is not None:
                last_gen_loss = gen_loss
            elif last_gen_loss is not None:
                gen_loss = last_gen_loss
            
            progress_bar.update(1)
                                     
        logger.info('Epoch [%d/%d], Discriminator Loss: %.8f, Generator Loss: %.8f',
                    epoch+1, config['train']['num_epochs'], disc_loss.item(), gen_loss.item())
        wandb.log({"Discriminator_loss": disc_loss, "Generator_loss": gen_loss})
        
        ### Save the model ###
        if epoch % 5 == 0:   
            model_file_name = f'./epoch_{epoch}.pth'
            torch.save(generator, os.path.join(full_name, model_file_name))                    
    
if __name__ == "__main__":
    main('./QuantGAN/configs/config.yaml')