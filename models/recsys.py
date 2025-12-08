import torch
from torch import nn
from torch import optim

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(features)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity 
        out = self.relu(out)
        return out

class RecSys(nn.Module):
    def __init__(self, total_users, total_titles, user_emb_dim: int = 32, title_emb_dim: int = 32):
        super(RecSys, self).__init__()
        self.usr_emb = nn.Embedding(total_users, user_emb_dim)
        self.title_emb = nn.Embedding(total_titles, title_emb_dim)
        self.fc1 = nn.Linear((user_emb_dim + title_emb_dim), 128)
        self.res_block1 = ResidualBlock(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.res_block2 = ResidualBlock(64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        user = x[:, 1].long()
        title = x[:, 0].long()

        u = self.usr_emb(user)
        t = self.title_emb(title)
        
        x = torch.concat([u, t], dim=1)
        x = self.fc1(x)
        x = self.res_block1(x)
        x = self.fc2(x)
        x = self.res_block2(x)
        x = self.fc3(x)
        return x