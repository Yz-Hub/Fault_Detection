# File: losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai import losses

class FocalLoss3D(nn.Module):
    """3D Focal Loss for class imbalance"""
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Automatically handle sigmoid activation
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss

class DiceLoss3D(nn.Module):
    """3D Dice Loss with MONAI implementation"""
    def __init__(self):
        super().__init__()
        self.dice_loss = losses.DiceLoss(
            to_onehot_y=False, 
            sigmoid=True,
            smooth_nr=1e-6,
            smooth_dr=1e-6
        )

    def forward(self, inputs, targets):
        return self.dice_loss(inputs, targets)

class DiceFocalLoss3DWithDS(nn.Module):
    """3D Dice-Focal Loss with Deep Supervision Support"""
    def __init__(self, dice_weight=0.6, focal_weight=0.4, 
                 ds_weights=[1.0, 0.5, 0.4, 0.3, 0.2], gamma=2.0, alpha=0.75):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ds_weights = ds_weights
        
        self.dice_loss = DiceLoss3D()
        self.focal_loss = FocalLoss3D(alpha=alpha, gamma=gamma)

    def _scale_target(self, target, size):
        """3D trilinear interpolation for target scaling"""
        return F.interpolate(target.float(), size=size, 
                           mode='trilinear', align_corners=False)

    def forward(self, inputs, targets):
        # Deep supervision handling (auto-detect)
        if isinstance(inputs, (list, tuple)):
            total_loss = 0.0
            main_pred = inputs[-1]
            
            # Deep supervision branches
            for i, pred in enumerate(inputs[:-1]):
                scaled_target = self._scale_target(targets, pred.shape[2:])
                dice = self.dice_loss(pred, scaled_target)
                focal = self.focal_loss(pred, scaled_target)
                total_loss += self.ds_weights[i] * (dice*self.dice_weight + focal*self.focal_weight)
            
            # Main branch
            main_dice = self.dice_loss(main_pred, targets)
            main_focal = self.focal_loss(main_pred, targets)
            main_loss = main_dice*self.dice_weight + main_focal*self.focal_weight
            
            return total_loss + main_loss
        else:
            # Single output mode
            dice = self.dice_loss(inputs, targets)
            focal = self.focal_loss(inputs, targets)
            return dice*self.dice_weight + focal*self.focal_weight

def build_loss_fn(config):
    if config.loss.name == "ce":
        return CrossEntropyLoss()
    elif config.loss.name == "bce":
        return BinaryCrossEntropyWithLogits()
    elif config.loss.name == "dice":
        return DiceLoss()
    elif config.loss.name == "dice_ce":
        return DiceCELoss()
    elif config.loss.name == "dice_focal":  # 保留原有实现
        return DiceFocalLoss()
    elif config.loss.name == "dice_focal3d":  # 新3D版本
        return DiceFocalLoss3DWithDS(
            dice_weight=getattr(config.loss, 'dice_weight', 0.6),
            focal_weight=getattr(config.loss, 'focal_weight', 0.4),
            ds_weights=getattr(config.loss, 'ds_weights', [1.0, 0.5, 0.4, 0.3, 0.2]),
            gamma=getattr(config.loss, 'gamma', 2.0),
            alpha=getattr(config.loss, 'alpha', 0.75)
        )
    else:
        raise ValueError("Unsupported loss type!")
