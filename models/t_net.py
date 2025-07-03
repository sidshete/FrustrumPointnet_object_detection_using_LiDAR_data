import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2 import PointNetFeat

class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        self.feat = PointNetFeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        # x: (B, 3, N)
        x = self.feat(x)  # (B, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (B, 3)
        return x
