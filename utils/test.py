# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np
# import os
# from models.frustrum_pointnet import FrustumPointNet
# from models.loss import compute_loss
# from datasets.frustrum_dataset import FrustumDataset
# #from utils.metrics import compute_3d_iou  # Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FrustumPointNetLoss(nn.Module):
#     def __init__(self, cls_weight=1.0, bbox_weight=1.0, iou_weight=0.5):
#         super(FrustumPointNetLoss, self).__init__()
#         self.cls_weight = cls_weight  # Weight for classification loss
#         self.bbox_weight = bbox_weight  # Weight for bounding box regression loss
#         self.iou_weight = iou_weight  # Weight for IoU loss (optional)

#     def forward(self, outputs, targets):
#         # Assuming `outputs` has 'cls' and 'bbox' keys for class predictions and bounding box predictions
#         cls_preds = outputs['cls']  # [batch_size, num_classes]
#         bbox_preds = outputs['bbox']  # [batch_size, 4] (x, y, z, w)

#         cls_labels = targets['cls']  # [batch_size]
#         bbox_labels = targets['bbox']  # [batch_size, 4]

#         # Classification Loss (Cross-Entropy)
#         cls_loss = F.cross_entropy(cls_preds, cls_labels)

#         # Bounding Box Loss (Smooth L1 Loss or MSE)
#         bbox_loss = F.smooth_l1_loss(bbox_preds, bbox_labels, reduction='mean')

#         # IoU Loss (if needed)
#         iou_loss = self.iou_loss(bbox_preds, bbox_labels)

#         # Combine losses with respective weights
#         total_loss = self.cls_weight * cls_loss + self.bbox_weight * bbox_loss + self.iou_weight * iou_loss
#         return total_loss

#     def iou_loss(self, pred_boxes, true_boxes):
#         """
#         Compute the IoU (Intersection over Union) loss for 3D bounding boxes.
#         IoU Loss = 1 - IoU(pred_boxes, true_boxes)
#         """

#         # Here you can compute the IoU based on the predicted and true bounding boxes
#         # This is a simple placeholder for 3D IoU computation; you can replace it with a more robust implementation.
#         iou = self.compute_3d_iou(pred_boxes, true_boxes)
#         return 1 - iou.mean()

#     def compute_3d_iou(self, pred_boxes, true_boxes):
#         """
#         Compute the 3D Intersection over Union (IoU) for bounding boxes.
#         This is a simplified version. For real-world applications, you can use more efficient libraries like `pyquaternion`.
#         """
#         # Placeholder function to calculate IoU between predicted and true 3D boxes
#         # You can replace this with a more efficient or robust implementation.
#         intersection = np.maximum(0, np.minimum(pred_boxes[:, 3], true_boxes[:, 3]) - np.maximum(pred_boxes[:, 0], true_boxes[:, 0]))  # Simplified 2D box overlap
#         union = (pred_boxes[:, 3] - pred_boxes[:, 0]) * (pred_boxes[:, 1] - pred_boxes[:, 1]) + (true_boxes[:, 3] - true_boxes[:, 0]) * (true_boxes[:, 1] - true_boxes[:, 1]) - intersection
#         return intersection / union

# # -------------------------------
# # Configuration
# # -------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define paths for the test data
# lidar_dir = 'data/lidar'
# label_dir = 'data/label'
# calib_dir = 'data/calib'

# # Define other parameters
# batch_size = 64

# # Initialize the dataset and dataloaders
# test_dataset = FrustumDataset(lidar_dir, label_dir, calib_dir, split='test')
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# # Load the checkpoint
# checkpoint_path = "checkpoints/Frustum_best_model.pth"
# checkpoint = torch.load(checkpoint_path)

# # Load the model's state dictionary from the checkpoint
# model = FrustumPointNet()
# model.load_state_dict(checkpoint)  # Use correct key if needed

# # Set the model to evaluation mode
# model.eval()

# # Start evaluating the model
# total_loss = 0.0
# total_samples = 0

# with torch.no_grad():
#     for batch_idx, (lidar_data, label_data) in enumerate(test_loader):
#         # Move data to the correct device (e.g., GPU if available)
#         lidar_data = lidar_data.to(device)  # Use your device (e.g., 'cuda' or 'cpu')
#         label_data = label_data.to(device)
        
#         # Forward pass
#         output = model(lidar_data)  # Make sure the input shape matches what your model expects
        
#         # Compute loss (use the appropriate loss function, for example)
#         loss = compute_loss(output, label_data)  # Define your loss computation
        
#         # Update total loss
#         total_loss += loss.item() * lidar_data.size(0)
#         total_samples += lidar_data.size(0)

# # Calculate average loss
# average_loss = total_loss / total_samples
# print(f"Test Loss: {average_loss}")


# test.py
import torch
from torch.utils.data import DataLoader, random_split
from models.frustrum_pointnet import FrustumPointNet
from datasets.frustrum_dataset import FrustumDataset
from models.loss import compute_loss  # your existing loss function
import numpy as np

def compute_segmentation_metrics(seg_pred, seg_label, num_classes=2):
    seg_pred_np = seg_pred.cpu().numpy()
    seg_label_np = seg_label.cpu().numpy()

    total_correct = (seg_pred_np == seg_label_np).sum()
    total_points = seg_label_np.size
    accuracy = total_correct / total_points

    ious = []
    for cls in range(num_classes):
        pred_cls = (seg_pred_np == cls)
        label_cls = (seg_label_np == cls)
        intersection = (pred_cls & label_cls).sum()
        union = (pred_cls | label_cls).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)

    mean_iou = np.nanmean(ious)
    return accuracy, mean_iou


def evaluate(model, loader, device='cuda'):
    model.eval()
    total_loss = 0
    accuracies = []
    ious = []

    with torch.no_grad():
        for batch in loader:
            pc = batch['point_cloud'].to(device).float()
            seg_label = batch['seg_label'].to(device).long()
            center_label = batch['center_label'].to(device).float()
            box_label = torch.cat([
                batch['center_label'],
                batch['size_label'],
                batch['angle_label'].unsqueeze(1)
            ], dim=1).to(device).float()

            seg_pred_logits, center_pred, box_pred = model(pc)
            loss, loss_dict = compute_loss(seg_pred_logits, seg_label, center_pred, box_pred, box_label, pc)
            total_loss += loss.item()

            # Get predicted segmentation labels (argmax over classes)
            seg_pred = seg_pred_logits.argmax(dim=2)

            # Compute metrics
            accuracy, mean_iou = compute_segmentation_metrics(seg_pred, seg_label, num_classes=2)
            accuracies.append(accuracy)
            ious.append(mean_iou)

    avg_loss = total_loss / len(loader)
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_iou = sum(ious) / len(ious)

    print(f"✅ Test Loss: {avg_loss:.4f}")
    print(f"✅ Test Accuracy: {avg_accuracy:.4f}")
    print(f"✅ Test Mean IoU: {avg_iou:.4f}")

    return avg_loss, avg_accuracy, avg_iou


def main():

    # Set paths and params
    lidar_dir = 'data/lidar'
    label_dir = 'data/label'
    calib_dir = 'data/calib'
    split = 'train'
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset split
    dataset = FrustumDataset(lidar_dir, label_dir, calib_dir, split)
    test_size = int(0.15 * len(dataset))
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size - test_size
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load model
    model = FrustumPointNet().to(device)
    model.load_state_dict(torch.load("checkpoints/Frustum_best_model.pth"))
    print("✅ Loaded pretrained model.")

    # Evaluate
    evaluate(model, test_loader, device=device)


if __name__ == "__main__":
    main()

