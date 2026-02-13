import torch
import torch.nn as nn
from jaxtyping import Float, Bool, Int
from torch import Tensor
from einops import einsum, rearrange
from .scaled_dot_product_attention import scaled_dot_product_attention
from .rope import RotaryPositionalEmbedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .multi_head_self_attention import MultiHeadSelfAttention
from .positionwise_feedforward import PositionwiseFeedForward

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rmsnorm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        self.rmsnorm2 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)


    def forward(
        self,
        in_features: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        # RMSNorm first
        x: Float[Tensor, "batch seq_len d_model"] = self.rmsnorm1(in_features)
        # Causal Multi-Head Self-Attention with RoPE
        token_positions: Int[Tensor, "batch seq_len"] = torch.arange(in_features.shape[1], device=in_features.device)[None, :]
        attn_output: Float[Tensor, "batch seq_len d_model"] = self.self_attn(x, token_positions=token_positions, theta=self.theta, max_seq_len=self.max_seq_len)
        # Residual connection + RMSNorm
        x = self.rmsnorm2(in_features + attn_output)
        # Position-wise Feedforward Network
        ffn_output: Float[Tensor, "batch seq_len d_model"] = self.ffn(x)
        # Residual connection
        output: Float[Tensor, "batch seq_len d_model"] = in_features + attn_output + ffn_output
        return output

