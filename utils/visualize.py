import torch
import open3d as o3d
import numpy as np
from models.frustrum_pointnet import FrustumPointNet
from datasets.frustrum_dataset import FrustumDataset
from torch.utils.data import DataLoader


def visualize_point_cloud(points, labels=None, title=""):
    """
    Visualize point cloud with optional semantic segmentation labels.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    if labels is not None:
        # Color based on labels (segmentation prediction or GT)
        colormap = np.array([
            [1, 0, 0],  # class 0 - red
            [0, 1, 0],  # class 1 - green
        ])
        colors = colormap[labels % len(colormap)]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    o3d.visualization.draw_geometries([pcd], window_name=title)


def estimate_mean_size():
    dataset = FrustumDataset('data/lidar', 'data/label', 'data/calib', split='train')
    loader = DataLoader(dataset, batch_size=64)
    all_sizes = []
    for batch in loader:
        sizes = batch['size_label']  # [B, 3]
        all_sizes.append(sizes)
    all_sizes = torch.cat(all_sizes, dim=0)
    mean_size = all_sizes.mean(dim=0)
    print(f"✅ Estimated MEAN_SIZE: {mean_size.tolist()}")
    return mean_size


def create_bbox(center, size, angle):
    """Create Open3D box mesh given center, size, and heading angle."""
    box = o3d.geometry.OrientedBoundingBox()
    box.center = center

    # Create a rotation matrix around Y-axis (assuming KITTI's convention)
    R = box.get_rotation_matrix_from_axis_angle([0, angle, 0])
    box.R = R
    box.extent = size  # size is already in absolute scale
    return box


def visualize_with_bboxes(points, pred_center, pred_size, pred_angle, gt_center, gt_size, gt_angle, title=""):
    """
    Visualize point cloud with both predicted and GT bounding boxes.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.paint_uniform_color([0.6, 0.6, 0.6])  # gray

    # Predicted bounding box (blue)
    pred_box = create_bbox(pred_center, pred_size, pred_angle)
    pred_box.color = [0, 0, 1]

    # Ground truth bounding box (red)
    gt_box = create_bbox(gt_center, gt_size, gt_angle)
    gt_box.color = [1, 0, 0]

    o3d.visualization.draw_geometries([pcd, pred_box, gt_box], window_name=title)


def run_inference(model, dataloader, mean_size, device="cuda"):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            pc = batch['point_cloud'].to(device).float()
            seg_pred, center_pred, box_pred = model(pc)

            # Segmentation visualization (optional)
            seg_label = batch['seg_label'].cpu().numpy()
            pred_label = seg_pred.argmax(dim=2).cpu().numpy()

            center_label = batch['center_label'].cpu().numpy()
            size_label = batch['size_label'].cpu().numpy()
            angle_label = batch['angle_label'].cpu().numpy()

            for i in range(min(len(pc), 2)):  # visualize up to 2 samples
                pts = pc[i].cpu().numpy()

                # Unnormalize prediction
                pred_center = center_pred[i].cpu().numpy()
                pred_size = box_pred[i, 3:6].cpu().numpy() * mean_size.numpy()
                pred_angle = box_pred[i, 6].cpu().numpy()

                # Ground truth
                gt_center = center_label[i]
                gt_size = size_label[i]
                gt_angle = angle_label[i]

                print(f"🔹 Pred center: {pred_center}, size: {pred_size}, angle: {pred_angle}")
                print(f"🔸 GT center:   {gt_center}, size: {gt_size}, angle: {gt_angle}")

                visualize_with_bboxes(
                    points=pts,
                    pred_center=pred_center,
                    pred_size=pred_size,
                    pred_angle=pred_angle,
                    gt_center=gt_center,
                    gt_size=gt_size,
                    gt_angle=gt_angle,
                    title="Prediction vs Ground Truth"
                )
            break  # just one batch


def main():
    # Load dataset
    val_dataset = FrustumDataset('data/lidar', 'data/label', 'data/calib', split='val')
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Load model
    model = FrustumPointNet().to("cuda")
    model.load_state_dict(torch.load("checkpoints/Frustum_best_model.pth"))
    print("✅ Model loaded for inference.")

    # Estimate mean size from training data
    mean_size = estimate_mean_size()

    # Run inference + visualization
    run_inference(model, val_loader, mean_size)


if __name__ == "__main__":
    main()
