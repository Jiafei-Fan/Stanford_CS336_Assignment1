import torch
import torch.nn as nn
from jaxtyping import Float, Bool
from torch import Tensor
from einops import einsum
from .softmax import softmax

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # 1st matmul: score
    qk: Float[Tensor, "... queries keys"] = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    sqrt_dk = Q.shape[-1] ** 0.5
    scaled_qk = qk / sqrt_dk
    if mask is not None:
        scaled_qk = torch.where(mask, scaled_qk, torch.tensor(float("-inf"), device=scaled_qk.device, dtype=scaled_qk.dtype))
    softmax_qk: Float[Tensor, "... queries keys"] = softmax(scaled_qk, dim=-1)
    # 2nd matmul: score and value
    output: Float[Tensor, "... queries d_v"] = einsum(softmax_qk, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return output