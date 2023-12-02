### DATA API - Time series examples ###

#pip install yfinance
import sys
import yfinance as yf

start = "2009-05-01"
end = "2018-12-31"

SP500_daily = yf.download("^GSPC", start, end)['Close']
HANGSENG_daily = yf.download("^HSI", start, end)['Close']
ShanghaiSE_daily = yf.download("000001.SS", start, end)['Close']
KOSPI_daily = yf.download("^KS11", start, end)['Close']

print(f"SP500_daily.shape: {SP500_daily.shape}")
print(f"HANGSENG_daily.shape: {HANGSENG_daily.shape}")
print(f"ShanghaiSE_daily.shape: {ShanghaiSE_daily.shape}")
print(f"KOSPI_daily.shape: {KOSPI_daily.shape}")

SP500_daily.to_csv(f'./QuantGAN/data/SP500_daily.csv')
HANGSENG_daily.to_csv(f'./QuantGAN/data/HANGSENG_daily.csv')
ShanghaiSE_daily.to_csv(f'./QuantGAN/data/ShanghaiSE_daily.csv')
KOSPI_daily.to_csv(f'./QuantGAN/data/KOSPI_daily.csv')