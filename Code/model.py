import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        if padding == 0:
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.02)
        self.conv2.weight.data.normal_(0, 0.02)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.02)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Generator(nn.Module):
    """Generator: 3 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """ 
    def __init__(self):
        super(Generator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(5, 100, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(100, 100, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(100, 5, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):        
        for layer in self.tcn:
            x = layer(x)            
        x = self.last(x)
        return x


class Discriminator(nn.Module):
    """Discrimnator: 1 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """ 
    def __init__(self, seq_len, conv_dropout=0.05):
        super(Discriminator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(5, 100, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(100, 100, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(100, 1, kernel_size=1, dilation=1)        
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())
        #self.to_prob = nn.Sequential(nn.Linear(seq_len, 1))

    def forward(self, x):
        for layer in self.tcn:
            x = layer(x)            
        x = self.last(x)
        return self.to_prob(x).squeeze()




class LSTMGenerator(nn.Module):
    """Generator with LSTM"""
    def __init__(self, latent_dim, ts_dim, hidden_dim=64, num_layers=1):
        super(LSTMGenerator, self).__init__()

        self.ts_dim = ts_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, ts_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.linear(out)
        out = out.permute(0, 2, 1)
        return out
    
# Discriminator with LSTM
class LSTMDiscriminator(nn.Module):
    def __init__(self, ts_dim, seq_len, hidden_dim=64, num_layers=1):
        super(LSTMDiscriminator, self).__init__()        
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(ts_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())
    def forward(self, x):
        x = x.permute(0, 2, 1)            
        out, _ = self.lstm(x)
        out = self.linear(out).squeeze()
        out = self.to_prob(out)
        return out
