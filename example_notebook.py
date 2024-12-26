import os
import yaml
import copy
import pickle
import ml_collections
import wandb, signatory

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import path as pt
from tqdm import tqdm
sns.set_style("darkgrid")  # 원하는 스타일 선택

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils import *
from src.baselines.networks.discriminators import UserDiscriminator
from src.baselines.networks.generators import UserGenerator
from src.baselines.trainer import *

from scipy.stats import entropy, wasserstein_distance, ks_2samp, spearmanr, kendalltau
from statsmodels.tsa.stattools import acf
from scipy.spatial.distance import jensenshannon
print(os.getcwd())
# Load configuration dict
config_dir = 'configs/config.yaml'
with open(config_dir) as file:
    config = ml_collections.ConfigDict(yaml.safe_load(file))

if (config.device == "cuda" and torch.cuda.is_available()):
    config.update({"device": "cuda:0"}, allow_val_change=True)
    torch.cuda.set_device(0)
else:
    config.update({"device": "cpu"}, allow_val_change=True)

set_seed(config.seed)
def rolling_window(data, window_size):
    n_windows = data.shape[0] - window_size + 1
    windows = np.zeros((n_windows, window_size, data.shape[1]))
    for idx in range(n_windows):
        windows[idx] = data[idx:idx + window_size]
    return windows

def compute_avg_emd(real_data, fake_data, window):
    emd = 0
    for i in range(real_data.shape[1]):  # Iterate over features
        real_dist = rolling_window(real_data[:, i, :].T, window).sum(axis=1).ravel()
        fake_dist = rolling_window(fake_data[:, i, :].T, window).sum(axis=1).ravel()
        emd += wasserstein_distance(real_dist, fake_dist)
    return emd


assets = ['AAPL', 'AMZN', 'BRK', 'GE', 'LIN', 'LLY', 'NEE', 'PLD', 'VZ', 'WMT', 'XOM']
sector = 'stock'
year = 2022
        
for asset in assets:
    config.update({"file_name": asset}, allow_val_change=True)

    # Step 1: Load and preprocess data
    df = pd.read_csv(f"./data/{sector}/{config.file_name}.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date', inplace=True)
    df = df.apply(pd.to_numeric).astype(float)

    # Step 2: Compute log returns
    log_returns = np.diff(np.log(df), axis=0)
    print(log_returns.shape)

    # Step 3: Scale the log returns
    log_returns_scaled, scalers = scaling(log_returns)

    # Step 4: Prepare initial prices and create rolling windows
    init_price = torch.from_numpy(np.array(df)[:-(config.n_steps), :]).float().unsqueeze(1)
    log_returns_scaled = torch.from_numpy(rolling_window(log_returns_scaled, config.n_steps)).float()
    print('init_price:', init_price.shape)
    print('log_returns_scaled:', log_returns_scaled.shape)

    perm_idx = torch.randperm(log_returns_scaled.shape[0])
    train_size = int(1.0*log_returns_scaled.shape[0])

    training_data = log_returns_scaled[perm_idx[:train_size]]
    train_init_price = init_price[perm_idx[:train_size]]
    print("training_data: ", training_data.shape)

    training_set = TensorDataset(training_data)
    train_dl = DataLoader(training_set, batch_size=config.batch_size, shuffle=True)

    generator = UserGenerator(config)
    discriminator = UserDiscriminator(config)
    trainer = GANTrainer(G=generator, D=discriminator, train_dl=train_dl, config=config)

    # Model training
    trainer.fit()

    full_name = "indices_300_128_G_lr_0.0002_D_lr_0.0001_noise_6_hidden_dim_100"

    """ Find the best epochs based on 100 days cumulative distribution """
    window = 100
    gen_tmp = UserGenerator(config)
    min_dist = float('inf')  
    best_epoch = 0

    

    batch_size = 1000
    for epoch in range(80, 200, 5):
        
        # Load generator for the current epoch
        generator.load_state_dict(torch.load(f'./results/models/{full_name}/Generator_{epoch}.pt'))
        generator.to(config.device)
        generator.eval()
        
        # Generate fake data
        noise = torch.randn(batch_size, config.noise_dim, config.n_steps, device=config.device)
        with torch.no_grad():        
            fake= generator(noise)                
            
        # Inverse scaling for real and fake data
        fake_data = inverse_scaling(fake, scalers)        
        real_data = inverse_scaling(training_data.transpose(1, 2), scalers)
        
        # Compute EMD
        emd = compute_avg_emd(real_data, fake_data, window)

        # Update best epoch if current EMD is lower
        if emd < min_dist:
            min_dist = emd
            best_epoch = epoch
            print(f"min_distance: {min_dist:.3f}, best_epoch: {best_epoch}") 

    generator.load_state_dict(torch.load(f'./results/models/{full_name}/Generator_{best_epoch}.pt'))
    generator.to(config.device)
    generator.eval()

    """ Fake data와 Test data를 비교 """
    batch_size = 2000
    noise = torch.randn(batch_size, config.noise_dim, config.n_steps, device=config.device)
    with torch.no_grad():
        fake = generator(noise)

    fake_data = inverse_scaling(fake, scalers)
    real_data = inverse_scaling(training_data.transpose(1, 2), scalers)

    # 각 feature의 전체 최소값 및 최대값 계산
    min_vals = real_data.min(axis=(0, 2), keepdims=True)  # Shape: (1, 5, 1)
    max_vals = real_data.max(axis=(0, 2), keepdims=True)  # Shape: (1, 5, 1)

    # 마스크 계산 (모든 샘플이 범위 내에 있는지 확인)
    mask = np.all((fake_data >= min_vals * 1.5) & (fake_data <= max_vals * 1.5), axis=(1, 2))

    # 마스크를 적용하여 fake_data 필터링
    print(real_data.shape, type(real_data))
    print(fake_data.shape, type(fake_data))
    fake_data = fake_data[mask]
    print(f"Filtered shape: {fake_data.shape}")


    """ Visualize the distribution of the real and fake data """
    fake_list = [fake_data[:, i, :] for i in range(fake_data.shape[1])]
    real_list = [real_data[:, i, :] for i in range(real_data.shape[1])]

    # Plot the distribution of the real and fake data
    windows = [1, 5, 20, 100]
    for j in range(config.n_vars):
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(28, 6))  

        for i in range(len(windows)):
            col = i

            real_dist = rolling_window(real_list[j].T, windows[i]).sum(axis=1).ravel()
            fake_dist = rolling_window(fake_list[j].T, windows[i]).sum(axis=1).ravel()        
            
            min_val = real_dist.min()
            max_val = real_dist.max()
            
            bins = np.linspace(min_val, max_val, 81)  
            
            sns.histplot(real_dist, bins=bins, kde=False, ax=axs[col], color='tab:blue', alpha=0.5, stat='density')
            sns.histplot(fake_dist, bins=bins, kde=False, ax=axs[col], color='tab:orange', alpha=0.5, stat='density')

            axs[col].set_xlim(*np.quantile(real_dist, [0.001, .999]))
            
            axs[col].set_title('{} day return distribution'.format(windows[i]), size=18)
            axs[col].yaxis.grid(True, alpha=0.5)
            axs[col].set_xlabel('Cumulative log return', fontsize=12)
            axs[col].set_ylabel('Frequency', fontsize=12)

        axs[0].legend(['Historical returns', 'Synthetic returns'])

    plt.show()