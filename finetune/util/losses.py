import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedSmoothL1LossWeighted(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask, loss_type='l1'):
        # prediction (b, 10*6)W
        # target (b, 10*6)
        # mask (b, 10*6)
        instance_weight = (target * mask).sum(-1, keepdims=True) / (mask.sum(-1, keepdims=True) + 1e-6) # mean of each sample
        instance_weight = instance_weight / instance_weight.sum() * prediction.shape[0] # re-distribute B weights
        # Compute the loss using a standard loss function (e.g., MSE)
        if loss_type == 'l1':
            loss = F.smooth_l1_loss(prediction, target, reduction="none") #(b, 10*6)
        elif loss_type == 'l2':
            loss = F.mse_loss(prediction, target, reduction="none")
        # Set the gradient to zero for labels equal to 0
        loss *= mask  # Element-wise multiplication to zero out gradients for target == 0
        
        loss *= instance_weight
        loss = loss.mean()
        return loss
    


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self):
        super(MaskedSmoothL1Loss, self).__init__()

    def forward(self, prediction, target, mask, loss_type='l1'):
        # prediction (b, 10*6)W
        # target (b, 10*6)
        # mask (b, 10*6)
        # Compute the loss using a standard loss function (e.g., MSE)
        if loss_type == 'l1':
            loss = F.smooth_l1_loss(prediction, target, reduction="none") #(b, 10*6)
        elif loss_type == 'l2':
            loss = F.mse_loss(prediction, target, reduction="none")
        # Set the gradient to zero for labels equal to 0
        loss *= mask  # Element-wise multiplication to zero out gradients for target == 0
        loss = loss.mean()
        return loss

class MaskedSmoothL1LossEqual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask):
        # prediction (b, 10*6)W
        # target (b, 10*6)
        # mask (b, 10*6)
        # Compute the loss using a standard loss function (e.g., MSE)
        loss = F.smooth_l1_loss(prediction, target, reduction="none") #(b, 10*6)

        # Set the gradient to zero for labels equal to 0
        loss *= mask  # Element-wise multiplication to zero out gradients for target == 0
        loss_mean_channel = loss.sum(-1) / (mask.sum(-1) + 1e-6)
        loss = loss_mean_channel.mean()
        return loss

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, prediction, target, mask):
        # prediction (b, 10*6, Bins)
        # target (b, 10*6)
        # mask (b, 10*6)

        B, N, C = prediction.size()
        loss = F.cross_entropy(prediction.view(-1, C), target.view(B * N), reduction="none")  # (B * N)

        # Set the gradient to zero for labels equal to 0
        loss *= mask.view(B * N)  # Element-wise multiplication to zero out gradients for target == 0

        loss = loss.mean()
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, prediction, target, mask):
        # prediction (b, 10*6)
        # target (b, 10*6)
        # mask (b, 10*6)
        # Compute the loss using a standard loss function (e.g., MSE)
        loss = F.mse_loss(prediction, target, reduction="none") #(b, 10*6)

        # Set the gradient to zero for labels equal to 0
        loss *= mask  # Element-wise multiplication to zero out gradients for target == 0
        
        loss = loss.mean()
        return loss


class Masked_L1Smooth_MSELoss(nn.Module):
    def __init__(self):
        super(Masked_L1Smooth_MSELoss, self).__init__()

    def forward(self, prediction, target, mask):
        # prediction (b, 10*6)
        # target (b, 10*6)
        # mask (b, 10*6)
        # Compute the loss using a standard loss function (e.g., MSE)
        prediction_first3, prediction_last3 = prediction.chunk(2, dim=1)  # (b, 30), (b, 30)
        target_first3, target_last3 = target.chunk(2, dim=1)  # (b, 30), (b, 30)

        loss_first3 = F.smooth_l1_loss(target_first3, prediction_first3, reduction="none") # (b, 30)
        loss_last3 = F.mse_loss(target_last3, prediction_last3, reduction="none") # (b, 30)

        loss = torch.concat([loss_first3, loss_last3], dim=1)

        # Set the gradient to zero for labels equal to 0
        loss *= mask  # Element-wise multiplication to zero out gradients for target == 0
        
        loss = loss.mean()
        return loss