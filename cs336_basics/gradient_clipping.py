import torch
from torch import nn

"""
During training, we can sometimes hit training examples that yield large gradients, 
which can destabilize training. 
To mitigate this, one technique often employed in practice is gradient clipping.
The idea is to enforce a limit on the norm of the gradient after each backward pass before taking an optimizer step.
"""
def gradient_clipping(parameters: list[nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6

    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return

    total_sq_norm = 0.0
    for p in params_with_grad:
        total_sq_norm += torch.sum(p.grad * p.grad).item()

    total_norm = total_sq_norm ** 0.5
    clip_coef = max_l2_norm / (total_norm + eps)

    if clip_coef < 1.0:
        for p in params_with_grad:
            p.grad.mul_(clip_coef)