import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs.view(-1, 2), targets.view(-1))
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def compute_loss(
    seg_pred,
    seg_label,
    center_delta,
    box_pred,
    box_gt,
    pc,  # NEW: point cloud input (B, N, 3)
    w_seg=1.0,
    w_center=0.001,
    w_box=0.001,
    normalize_box=True,
    use_log_center_loss=True,
    dynamic_center_weight=True,
    current_epoch=None,
    total_epochs=300,
    loss_type='mse'
):
    """
    Compute the multi-task loss for 3D object detection.
    Args:
        seg_pred: (B, N, 2)
        seg_label: (B, N)
        center_delta: (B, 3)
        box_pred: (B, 7)
        box_gt: (B, 7)
        pc: (B, N, 3)
    Returns:
        total_loss: scalar
        loss_dict: dictionary with individual loss components
    """

    # 1. Segmentation Loss
    criterion_seg = FocalLoss(alpha=1.0, gamma=2.0)
    loss_seg = criterion_seg(seg_pred.view(-1, 2), seg_label.view(-1))

    # 2. Center Offset Loss (TNet style)
    center_gt = box_gt[:, :3]              # absolute GT center
    pc_mean = torch.mean(pc, dim=1)        # (B, 3)
    offset_gt = center_gt - pc_mean  
    #print("center_delta stats — mean:", center_delta.mean().item(), "std:", center_delta.std().item())
    #print("offset_gt stats — mean:", offset_gt.mean().item(), "std:", offset_gt.std().item())  # what TNet is predicting
    if loss_type == 'smoothl1':
        criterion_center = nn.SmoothL1Loss(beta=1.0)
    elif loss_type == 'l1':
        criterion_center = nn.L1Loss()
    else:
        criterion_center = nn.MSELoss()

    # Normalize to unit vectors per sample to avoid large scale discrepancies
    offset_gt = offset_gt / offset_gt.norm(dim=1, keepdim=True).clamp(min=1e-6)
    center_delta = center_delta / center_delta.norm(dim=1, keepdim=True).clamp(min=1e-6)

    raw_center_loss = criterion_center(center_delta, offset_gt)

    loss_center = torch.log(1 + raw_center_loss) if use_log_center_loss else raw_center_loss
    # cos_sim = F.cosine_similarity(center_delta, offset_gt, dim=1)
    # loss_center = 1 - cos_sim.mean()  # optional

    # Optional dynamic scaling of center weight
    if dynamic_center_weight and current_epoch is not None:
        w_center = max(w_center * (1 - current_epoch / total_epochs), 0.05)

    # 3. Normalize Box Predictions (optional)
    if normalize_box:
        box_pred = box_pred - box_pred.mean(dim=0, keepdim=True)
        box_gt = box_gt - box_gt.mean(dim=0, keepdim=True)

    # 4. Box Regression Loss
    criterion_box = nn.SmoothL1Loss(beta=1.0)
    loss_box = criterion_box(box_pred, box_gt)

    # 5. Total Loss
    total_loss = w_seg * loss_seg + w_center * loss_center + w_box * loss_box

    return total_loss, {
        'seg': loss_seg.item(),
        'center': loss_center.item(),
        'box': loss_box.item(),
        'total': total_loss.item(),
        'w_center': w_center
    }

