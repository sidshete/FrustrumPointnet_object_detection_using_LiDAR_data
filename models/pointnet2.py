import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetFeat(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetFeat, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.global_feat = global_feat

    def forward(self, x):
        # x: (B, 3, N)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # (B, 1024, N)
        x_max = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        if self.global_feat:
            return x_max.view(-1, 1024)
        else:
            x_max_repeated = x_max.repeat(1, 1, x.shape[2])  # (B, 1024, N)
            return torch.cat([x, x_max_repeated], 1)  # (B, 1024+1024, N)
