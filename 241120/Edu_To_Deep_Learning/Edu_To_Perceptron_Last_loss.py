import torch
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [0.0], [0.0], [1.0]])

model = torch.nn.Linear(2, 1)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = torch.sigmoid(model(x))
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

print(loss.item())