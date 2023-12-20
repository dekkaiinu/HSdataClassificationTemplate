import torch
import torch.nn as nn

class MLP_BatchNotm(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.1):
        super(MLP_BatchNotm, self).__init__()
        # self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        # self.dropout1 = nn.Dropout(p=dropout_prob / 6)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=dropout_prob / 5)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x = self.bn0(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        # x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x