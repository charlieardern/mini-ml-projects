from __future__ import annotations

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = nn.MSELoss()


def train_step(model, loss_fn, optimizer, scheduler, train_loader):
    model.to(device)
    model.train()
    for x, y in train_loader:
        x_dev, y_dev = x.to(device), y.to(device)
        y_pred = model(x_dev)
        loss = loss_fn(y_dev, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


def evaluate_model(model, loss_fn, train_loader, test_loader):
    model.to(device)
    model.eval()
    train_loss = 0.0
    test_loss = 0.0

    with torch.inference_mode():
        for x, y in train_loader:
            x_dev, y_dev = x.to(device), y.to(device)
            y_pred = model(x_dev)
            train_loss += loss_fn(y_dev, y_pred) / len(train_loader)
        for x, y in test_loader:
            x_dev, y_dev = x.to(device), y.to(device)
            y_pred = model(x_dev)
            test_loss += loss_fn(y_dev, y_pred) / len(test_loader)
    print(f"Train loss: {train_loss:.5f}, Test loss: {test_loss:.5f}")
    return train_loss, test_loss
