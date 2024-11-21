import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = torch.mean((y_pred - y_true) ** 2)  # MSE
        l2_reg = torch.sum(y_pred ** 2)  # L2 Regularization
        return mse_loss + 0.01 * l2_reg  # 조합

loss = CustomLoss()