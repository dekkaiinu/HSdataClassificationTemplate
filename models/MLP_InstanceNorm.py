import torch
import torch.nn as nn

class MLP_InstanceNorm(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.0):
        super(MLP_InstanceNorm, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.InstanceNorm1d(256)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.InstanceNorm1d(128)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.InstanceNorm1d(64)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x