import torch, joblib
import numpy as np
import random
from preprocess.gaussianize import *
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf

# 스케일링을 위한 함수 정의
def scaling(data, n_groups):
    columns = []    
    joblib.dump(data, f'./Dataset/log_returns_original.pkl')
    for i in range(n_groups):        
        standardScaler1 = StandardScaler()
        standardScaler2 = StandardScaler()
        gaussianize = Gaussianize()        
        log_returns = data[:, i].reshape(-1, 1)
        log_returns_preprocessed = standardScaler2.fit_transform(gaussianize.fit_transform(standardScaler1.fit_transform(log_returns)))        
        
        joblib.dump(standardScaler1, f'./Dataset/pickle/{i}_standardScaler1.pkl')
        joblib.dump(standardScaler2, f'./Dataset/pickle/{i}_standardScaler2.pkl')
        joblib.dump(gaussianize, f'./Dataset/pickle/{i}_gaussianize.pkl')    
        joblib.dump(log_returns, f'./Dataset/pickle/{i}_log_returns.pkl')                        
        columns.append(log_returns_preprocessed.reshape(-1))
        
    returns = np.array(columns).T
    joblib.dump(returns, f'./Dataset/log_returns_preprocessed.pkl')
    return returns

def inverse_process(y, asset_idx):
    standardScaler1 = joblib.load(f'./Dataset/pickle/{asset_idx}_standardScaler1.pkl')
    standardScaler2 = joblib.load(f'./Dataset/pickle/{asset_idx}_standardScaler2.pkl')
    gaussianize = joblib.load(f'./Dataset/pickle/{asset_idx}_gaussianize.pkl')
    log_returns = joblib.load(f'./Dataset/pickle/{asset_idx}_log_returns.pkl')
     
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    y = standardScaler2.inverse_transform(y)
    y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
    y = standardScaler1.inverse_transform(y)
    return y, log_returns

def compute_gradient_penalty(discriminator, real_samples, fake_samples, z, i):
    """Compute the gradient penalty for WGAN-GP for conditional GANs with time series data."""
    # real_samples, fake_samples는 시계열 데이터이며 torch Tensors입니다.
    
    # 알파(Alpha) 계수를 사용하여 진짜와 가짜 사이의 보간(interpolation)을 계산
    # 시계열 데이터 형태에 맞춰 alpha 차원을 조정
    alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device).expand_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # 조건 벡터를 보간된 샘플에 추가하여 critic에 전달
    critic_interpolates = discriminator(interpolates, z, i)
    
    # 그라디언트 계산
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=real_samples.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # 그라디언트의 노름(norm) 계산 (시계열 데이터 차원에 맞춰 수정)
    gradients_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    
    # 그라디언트 패널티 (1 - 그라디언트 노름)^2의 평균 계산
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # Ensure deterministic behavior on GPU (if this is desired)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False


