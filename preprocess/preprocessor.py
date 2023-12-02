import torch
import numpy as np
import pandas as pd
from preprocess.gaussianize import *
from preprocess.acf import *
import joblib
import os 

def processor(data, file_path ,file_name):
    df = data['Close']
    log_returns = np.log(df/df.shift(1))[1:].to_numpy().reshape(-1, 1)
    standardScaler1 = StandardScaler()
    standardScaler2 = StandardScaler()
    gaussianize = Gaussianize()
    log_returns_preprocessed = standardScaler2.fit_transform(gaussianize.fit_transform(standardScaler1.fit_transform(log_returns)))
    
    path = os.path.join(file_path, 'pickle')
    if not os.path.exists(path):
        os.makedirs(path)
    
    file_name = file_name.split('.')[0]
    
    joblib.dump(standardScaler1, os.path.join(file_path, f'{file_name}_standardScaler1.pkl'))
    joblib.dump(standardScaler2, os.path.join(file_path, f'{file_name}_standardScaler2.pkl'))
    joblib.dump(gaussianize, os.path.join(file_path, f'{file_name}_gaussianize.pkl'))
    joblib.dump(log_returns, os.path.join(file_path, f'{file_name}_log_returns.pkl'))
    joblib.dump(log_returns_preprocessed, os.path.join(file_path, f'{file_name}_log_returns_preprocessed.pkl'))
    
    return log_returns_preprocessed
