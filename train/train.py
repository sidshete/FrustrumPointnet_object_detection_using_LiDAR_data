import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.frustrum_pointnet import FrustumPointNet
from models.loss import compute_loss
from datasets.frustrum_dataset import FrustumDataset
from train.trainer import Trainer
from torch.utils.data import random_split



def main():
    # Set paths
    lidar_dir = 'data/lidar'
    label_dir = 'data/label'
    calib_dir = 'data/calib'
    split = 'train'
    batch_size = 64
    num_epochs = 300

    # Dataset and loader
    dataset = FrustumDataset(lidar_dir, label_dir, calib_dir, split)
    #seg_label_tensor = torch.tensor(dataset.sample_list[0]['seg_label'])
    #print("seg_label stats:", torch.unique(seg_label_tensor, return_counts=True))

    # New split sizes for train, validation, and test datasets (70-15-15 split)
    test_size = int(0.15 * len(dataset))
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size - test_size

    # Split dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Data loaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = FrustumPointNet().to('cuda')

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-6)

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, optimizer, compute_loss)
    trainer.train(num_epochs)

if __name__ == "__main__":
    main()
