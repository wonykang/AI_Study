import torch

criterion = torch.nn.BCELoss()
y_true = torch.tensor([[1.0]])
y_pred = torch.tensor([[0.9]])
loss = criterion(y_pred, y_true)
print(loss.item())