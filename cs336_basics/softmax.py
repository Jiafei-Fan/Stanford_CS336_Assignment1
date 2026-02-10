import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

def softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_vals = torch.max(in_features, dim=dim, keepdim=True).values
    exp_vals: Float[Tensor, "..."] = torch.exp(in_features - max_vals)
    sum_exp_vals = torch.sum(exp_vals, dim=dim, keepdim=True)
    softmax_vals = exp_vals / sum_exp_vals
    return softmax_vals

