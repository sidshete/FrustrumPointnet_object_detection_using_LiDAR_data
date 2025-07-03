import torch
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, scheduler=None, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=300
)


    def train(self, num_epochs):
        self.model.train()
        best_val_loss = float('inf')  # Initialize best loss

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_seg_loss = 0.0
            epoch_center_loss = 0.0
            epoch_box_loss = 0.0
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)

            for batch in loop:
                pc = batch['point_cloud'].to(self.device).float()
                seg_label = batch['seg_label'].to(self.device).long()
                center_label = batch['center_label'].to(self.device).float()
                box_label = torch.cat([
                    batch['center_label'], 
                    batch['size_label'], 
                    batch['angle_label'].unsqueeze(1)
                ], dim=1).to(self.device).float()

                self.optimizer.zero_grad()
                seg_pred, center_pred, box_pred = self.model(pc)
                loss, loss_dict = self.loss_fn(seg_pred, seg_label, center_pred, box_pred, box_label, pc)

                # Backprop and optimization
                loss.backward()
                self.optimizer.step()

                # Accumulate losses
                epoch_loss += loss.item()
                epoch_seg_loss += loss_dict['seg']
                epoch_center_loss += loss_dict['center']
                epoch_box_loss += loss_dict['box']

                loop.set_postfix(train_loss=loss.item())
                #print("seg_label dtype:", seg_label.dtype)
                #print("seg_label unique values:", torch.unique(seg_label))


            val_loss = self.validate(epoch, num_epochs)
            # Step the scheduler based on validation loss
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(self.model.state_dict(), f"checkpoints/Frustum_best_model.pth")
                tqdm.write(f"✅ Saved best model at epoch {epoch+1} — Val Loss improved to {val_loss:.4f}")

            # Log progress with individual losses
            tqdm.write(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {epoch_loss:.4f}, Seg Loss: {epoch_seg_loss:.4f}, "
                       f"Center Loss: {epoch_center_loss:.4f}, Box Loss: {epoch_box_loss:.4f}, Val Loss: {val_loss:.4f}")

    def validate(self, epoch, num_epochs):
        self.model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_center_loss = 0.0
        val_box_loss = 0.0
        loop = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)

        with torch.no_grad():
            for batch in loop:
                pc = batch['point_cloud'].to(self.device).float()
                seg_label = batch['seg_label'].to(self.device).long()
                center_label = batch['center_label'].to(self.device).float()
                box_label = torch.cat([
                    batch['center_label'], 
                    batch['size_label'], 
                    batch['angle_label'].unsqueeze(1)
                ], dim=1).to(self.device).float()

                seg_pred, center_pred, box_pred = self.model(pc)
                loss, loss_dict = self.loss_fn(seg_pred, seg_label, center_pred, box_pred, box_label, pc)
                #print("seg_pred min/max:", seg_pred.min().item(), seg_pred.max().item())

                val_loss += loss.item()
                val_seg_loss += loss_dict['seg']
                val_center_loss += loss_dict['center']
                val_box_loss += loss_dict['box']  # Correct the key to 'box'

                loop.set_postfix(val_loss=loss.item())

        self.model.train()
        tqdm.write(f"Validation Loss: {val_loss:.4f}, Seg Loss: {val_seg_loss:.4f}, "
                   f"Center Loss: {val_center_loss:.4f}, Box Loss: {val_box_loss:.4f}")
        return val_loss
