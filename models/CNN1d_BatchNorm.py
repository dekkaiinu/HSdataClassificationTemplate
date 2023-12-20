import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class CNN1d_BatchNorm(nn.Module):
    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    # def _get_final_flattened_size(self):
        # with torch.no_grad():
            # x = torch.zeros(1, 1, self.input_channels)
            # x = self.pool5(self.conv5(x))
        # return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(CNN1d_BatchNorm, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv1 = nn.Conv1d(1, 16, kernel_size)
        # self.ln1 = nn.LayerNorm((16, 149))
        self.bn1 = nn.BatchNorm1d((16))
        self.pool1 = nn.AvgPool1d(pool_size)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size)
        # self.ln2 = nn.LayerNorm((32, 72))
        self.bn2 = nn.BatchNorm1d((32))
        self.pool2 = nn.AvgPool1d(pool_size)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size)
        # self.ln3 = nn.LayerNorm((64, 34))
        self.bn3 = nn.BatchNorm1d((64))
        self.pool3 = nn.AvgPool1d(pool_size)

        self.conv4 = nn.Conv1d(64, 128, kernel_size)
        self.bn4 = nn.BatchNorm1d((128))
        # self.ln4 = nn.LayerNorm((128, 15))
        self.pool4 = nn.AvgPool1d(pool_size)
        # self.features_size = self._get_final_flattened_size()

        # [n4 is set to be 100]
        self.fc1 = nn.Linear(128 * 7, 128)
        # self.ln5 = nn.LayerNorm(128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = torch.relu(self.bn1(x))
        # print(x.shape)
        # exit()
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn5(self.fc1(x)))
        x = self.fc2(x)
        return x
