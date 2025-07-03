import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2 import PointNetFeat
from models.t_net import TNet

class FrustumPointNet(nn.Module):
    def __init__(self, num_classes=2):
        super(FrustumPointNet, self).__init__()

        # 1. Segmentation PointNet
        self.seg_feat = PointNetFeat(global_feat=False)
        self.seg_conv1 = nn.Conv1d(2048, 1024, 1)
        self.seg_conv2 = nn.Conv1d(1024, 512, 1)
        self.seg_conv3 = nn.Conv1d(512, 256, 1)
        self.seg_conv4 = nn.Conv1d(256, num_classes, 1)

        # 2. Center Regression (T-Net)
        self.tnet = TNet()

        # 3. Box Estimation Network
        self.box_fc1 = nn.Linear(2048, 512)
        self.box_fc2 = nn.Linear(512, 256)
        self.box_fc3 = nn.Linear(256, 7) 


    def forward(self, pc):
        B, N, _ = pc.shape
        pc = pc.transpose(2, 1).contiguous()  # (B, 3, N)

        # 1. Segmentation
        seg_feat = self.seg_feat(pc)  # (B, 2048, N)
        x = F.relu(self.seg_conv1(seg_feat))
        x = F.relu(self.seg_conv2(x))
        x = F.relu(self.seg_conv3(x))
        seg_pred = self.seg_conv4(x).transpose(2, 1).contiguous()  # (B, N, num_classes)    

        # 2. T-Net center regression
        center_delta = self.tnet(pc)  # (B, 3)

        # 3. Box estimation (only foreground points)
        global_feat = torch.max(seg_feat, 2)[0]  # (B, 2048)

        x = F.relu(self.box_fc1(global_feat))
        x = F.relu(self.box_fc2(x))
        box_pred = self.box_fc3(x)  # (B, 7)


        return seg_pred, center_delta, box_pred
