import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import einsum

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        """
        theta: float \theta value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # create/precompute sin_cache and cos_cache tensor with shape i*k where i is max_seq_len and k is d_kd2
        k_idx: Float[Tensor, "d_kd2"] = torch.arange(self.d_k // 2, device=device, dtype=torch.float32)
        k_idx: Float[Tensor, "d_kd2"] = 1 / (theta ** (2 * k_idx / self.d_k))
        position: Float[Tensor, "seq_len"] = torch.arange(max_seq_len, device=device)

        angle: Float[Tensor, "seq_len d_kd2"] = einsum(position, k_idx, "seq_len, d_kd2 -> seq_len d_kd2")
        self.register_buffer("sin_cache", angle.sin(), persistent=False)
        self.register_buffer("cos_cache", angle.cos(), persistent=False)

        
        
    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        """
        
        """
        # get the sin and cos values for the token positions from the cache
        sin: Float[Tensor, "... seq_len d_kd2"] = self.sin_cache[token_positions.long()]
        cos: Float[Tensor, "... seq_len d_kd2"] = self.cos_cache[token_positions.long()]

        # split the last dimension of x into even and odd parts
        x_even: Float[Tensor, "... seq_len d_kd2"] = x[..., ::2]
        x_odd: Float[Tensor, "... seq_len d_kd2"] = x[..., 1::2]

        # apply pairwise 2D rotation
        out_even: Float[Tensor, "... seq_len d_kd2"] = x_even * cos - x_odd * sin
        out_odd: Float[Tensor, "... seq_len d_kd2"] = x_even * sin + x_odd * cos

        # interleave back to (..., seq_len, d_k)
        out: Float[Tensor, "... seq_len d_k"] = torch.stack((out_even, out_odd), dim=-1).reshape(*x.shape)
        return out






