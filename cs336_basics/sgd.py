from collections.abc import Callable, Iterable
from typing import Optional
import math
import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    """
    {
        'state': {
                0: {'momentum_buffer': tensor(...), ...},
                1: {'momentum_buffer': tensor(...), ...},
                2: {'momentum_buffer': tensor(...), ...},
                3: {'momentum_buffer': tensor(...), ...}
            },
        'param_groups': [
            {
                'lr': 0.01,
                'weight_decay': 0,
                ...
                'params': [0]
                'param_names' ['param0']  (optional)
            },
            {
                'lr': 0.001,
                'weight_decay': 0.5,
                ...
                'params': [1, 2, 3]
                'param_names': ['param1', 'layer.weight', 'layer.bias'] (optional)
            }
        ]
    }
    """
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]          # Get state associated with p.
                t = state.get("t", 0)          # Get iteration number (default 0).
                grad = p.grad.data             # Get gradient of loss w.r.t. p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update in-place.
                state["t"] = t + 1             # Increment iteration number.

        return loss


# Minimal training loop example (same structure as the screenshot)
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()                 # Reset gradients
    loss = (weights ** 2).mean()    # Scalar loss
    print(loss.cpu().item())
    loss.backward()                 # Backward pass computes gradients
    opt.step()                      # Optimizer step

# Learning-rate tuning experiment (run 10 iters for each lr)
def run_lr(lr: float, iters: int = 10):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses = []
    for _ in range(iters):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return losses


for lr in [1e1, 1e2, 1e3]:
    losses = run_lr(lr, iters=10)
    print(f"lr={lr}: {losses}")