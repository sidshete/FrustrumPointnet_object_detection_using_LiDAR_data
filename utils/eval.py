import torch
import argparse
from torch.utils.data import DataLoader
from models.frustrum_pointnet import FrustumPointNet
from datasets.frustrum_dataset import FrustumDataset
from utils.visualize import visualize_prediction
from utils.metrics import compute_segmentation_accuracy, compute_center_distance

def evaluate(model_path, split='val'):
    # Load model
    model = FrustumPointNet()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    # Load dataset
    dataset = FrustumDataset('data/lidar', 'data/label', 'data/calib', split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_acc = 0.0
    total_dist = 0.0

    for i, batch in enumerate(dataloader):
        pc = batch['point_cloud'].cuda().float()
        seg_gt = batch['seg_label'].squeeze().numpy()
        center_gt = batch['center_label'].squeeze().numpy()

        with torch.no_grad():
            seg_pred, center_pred, box_pred = model(pc)

        seg_pred_np = seg_pred[0].cpu().numpy()
        center_pred_np = center_pred[0].cpu().numpy()
        box_pred_np = box_pred[0].cpu().numpy()

        # Metrics
        acc = compute_segmentation_accuracy(seg_pred_np, seg_gt)
        dist = compute_center_distance(center_pred_np, center_gt)
        total_acc += acc
        total_dist += dist

        # Visualize
        visualize_prediction(pc[0].cpu().numpy(), seg_pred_np, f"sample_{i}", center_pred_np, box_pred_np)
        print("GT center:", center_gt)
        print("Pred center:", center_pred_np)
        print(f"[{i+1}/{len(dataloader)}] Acc: {acc:.4f}, CenterDist: {dist:.2f}")

    avg_acc = total_acc / len(dataloader)
    avg_dist = total_dist / len(dataloader)
    print(f"\n✅ Evaluation done. Avg Seg Acc: {avg_acc:.4f}, Avg Center Dist: {avg_dist:.2f}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to evaluate on (val/test)')
    args = parser.parse_args()

    evaluate(args.model_path, args.split)