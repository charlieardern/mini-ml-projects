from __future__ import annotations

from timeit import default_timer

import torch
from datasets import test_data, train_data
from models import BasicUNet
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import evaluate_model, loss_fn, train_step

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device}")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_sampler=64, shuffle=False)

model = BasicUNet(in_channels=1, out_channels=1)
model = nn.DataParallel(model)
model = torch.compile(model)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
epochs = 10
num_time_steps = 1000


for epoch in tqdm(range(epochs)):
    t = torch.randint(0, num_time_steps, (1,)).item()
    eps = torch.rand(1).item()
    beta = torch.linspace(0.0004, 0.02, steps=num_time_steps)
    alpha_t = torch.prod(1 - beta[:t])

    model.to(device)
    model.train()
    for x_0, _ in train_loader:
        x_0 = x_0.to(device)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * eps
        eps_th = model(x_t)
        loss = loss_fn(eps_th, eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    mu_th = (x_t - beta[t] * eps_th(x_t) / torch.sqrt(1 - alpha_t)) / torch.sqrt(
        alpha_t
    )

    evaluate_model(model, loss_fn, train_loader, test_loader)
