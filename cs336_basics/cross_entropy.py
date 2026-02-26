import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange, einsum

def cross_entropy(inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 1) Subtract the largest element for numerical stability (per example)
    # m: (B, 1)
    m = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - m  # (B, V), max is now 0

    # 2) Cancel out log/exp: do NOT compute softmax then log.
    #    Instead compute logsumexp in a stable way:
    # logsumexp(inputs) = m + log(sum(exp(inputs - m)))
    # logsumexp: (B,)
    logsumexp = m.squeeze(-1) + torch.log(torch.exp(shifted).sum(dim=-1))

    # 3) Gather the logit of the true class o_y: (B,)
    # targets.unsqueeze(-1): (B, 1)
    true_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 4) Per-example loss: (B,)
    loss_per_example = -true_logits + logsumexp

    # 5) Average across the batch: scalar
    return loss_per_example.mean()
