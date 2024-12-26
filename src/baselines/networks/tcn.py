import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.utils.spectral_norm as spectral_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# class TimestepBatchNorm1d(nn.Module):
#     def __init__(self, num_features):
#         super(TimestepBatchNorm1d, self).__init__()
#         self.bn = nn.BatchNorm1d(num_features)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
#         x = self.bn(x)
#         x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
#         return x

# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1, spec_norm=False):
#         super(TemporalBlock, self).__init__()
        
#         # 첫 번째 합성곱 레이어
#         conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.conv1 = spectral_norm(conv1) if spec_norm else conv1
#         self.bn1 = nn.BatchNorm1d(n_outputs)  # Batch normalization 추가
#         self.chomp1 = Chomp1d(padding)
#         self.relu1 = nn.PReLU()
#         self.dropout1 = nn.Dropout(dropout)

#         # 두 번째 합성곱 레이어
#         conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.conv2 = spectral_norm(conv2) if spec_norm else conv2
#         self.bn2 = nn.BatchNorm1d(n_outputs)  # Batch normalization 추가
#         self.chomp2 = Chomp1d(padding)
#         self.relu2 = nn.PReLU()
#         self.dropout2 = nn.Dropout(dropout)

#         # 네트워크 구성
#         if padding == 0:
#             self.net = nn.Sequential(
#                 self.conv1, self.bn1, self.relu1, self.dropout1, 
#                 self.conv2, self.bn2, self.relu2, self.dropout2
#             )
#         else:
#             self.net = nn.Sequential(
#                 self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
#                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
#             )
        
#         self.downsample = None
#         if n_inputs != n_outputs:
#             self.downsample = nn.Sequential(
#                 nn.Conv1d(n_inputs, n_outputs, 1),
#                 nn.BatchNorm1d(n_outputs)  # Batch normalization
#             )
        
#         self.relu = nn.PReLU()
#         self.init_weights()

#     def init_weights(self):
#         nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
#         nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
#         if self.downsample is not None:
#             nn.init.kaiming_normal_(self.downsample[0].weight, mode='fan_out', nonlinearity='relu')
        
#         nn.init.constant_(self.relu1.weight, 0.25)
#         nn.init.constant_(self.relu2.weight, 0.25)
#         nn.init.constant_(self.relu.weight, 0.25)

#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return out, self.relu(out + res)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, spec_norm=False, n_steps=128, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # First convolutional block
        conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv1 = spectral_norm(conv1) if spec_norm else conv1
        self.chomp1 = None if padding == 0 else Chomp1d(padding)  # Apply Chomp1d only if padding > 0
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(n_outputs)

        # Second convolutional block
        conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = spectral_norm(conv2) if spec_norm else conv2
        self.chomp2 = None if padding == 0 else Chomp1d(padding)  # Apply Chomp1d only if padding > 0
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(n_outputs)

        # Residual connection
        self.downsample = None
        if n_inputs != n_outputs:
            self.downsample = nn.Sequential(
                nn.Conv1d(n_inputs, n_outputs, kernel_size=1),
                nn.BatchNorm1d(n_outputs)
            )
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample[0].weight, nonlinearity='leaky_relu')

    def apply_batch_norm(self, x, bn_layer):
        #x = x.transpose(1, 2)  # (batch_size, seq_length, n_outputs)
        x = bn_layer(x)        # Apply BatchNorm1d
        #x = x.transpose(1, 2)  # (batch_size, n_outputs, seq_length)
        return x

    def forward(self, x):
        # First convolution block
        out = self.conv1(x)
        if self.chomp1:  # Apply Chomp1d only if defined
            out = self.chomp1(out)
        out = self.apply_batch_norm(out, self.bn1)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second convolution block
        out = self.conv2(out)
        if self.chomp2:  # Apply Chomp1d only if defined
            out = self.chomp2(out)
        out = self.apply_batch_norm(out, self.bn2)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return out, self.relu(out + res)