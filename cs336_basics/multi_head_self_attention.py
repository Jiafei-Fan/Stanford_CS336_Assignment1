import torch
import torch.nn as nn
from jaxtyping import Float, Bool, Int
from torch import Tensor
from einops import einsum, rearrange
from .scaled_dot_product_attention import scaled_dot_product_attention
from .rope import RotaryPositionalEmbedding
from .linear import Linear

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int) -> None:
        """

        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # d_k = d_v = d_model // num_heads
        # d_in = d_model
        self.w_q: Linear = Linear(in_features=d_model, out_features=d_model)
        self.w_k: Linear = Linear(in_features=d_model, out_features=d_model)
        self.w_v: Linear = Linear(in_features=d_model, out_features=d_model)
        self.w_o: Linear = Linear(in_features=d_model, out_features=d_model)

    def forward(
        self,
        in_features: Float[Tensor, "... seq_len d_in"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        theta: float = 100000.0,
        max_seq_len: int = 2048
    ) -> Float[Tensor, "... seq_len d_in"]:
        """
        """
        q: Float[Tensor, "... seq_len d_model"] = self.w_q(in_features)
        k: Float[Tensor, "... seq_len d_model"] = self.w_k(in_features)
        v: Float[Tensor, "... seq_len d_model"] = self.w_v(in_features)
        # reshape for multi-head attention
        q_heads = rearrange(q, "... seq_len (head d_k) -> ... head seq_len d_k", head=self.num_heads)
        k_heads = rearrange(k, "... seq_len (head d_k) -> ... head seq_len d_k", head=self.num_heads)
        v_heads = rearrange(v, "... seq_len (head d_v) -> ... head seq_len d_v", head=self.num_heads)
        if token_positions is not None:
            rope = RotaryPositionalEmbedding(theta=theta, d_k=q_heads.shape[-1], max_seq_len=max_seq_len, device=in_features.device)
            q_heads = rope(q_heads, token_positions)
            k_heads = rope(k_heads, token_positions)
        # create mask for causal attention
        device = in_features.device
        seq_len = in_features.shape[-2]
        base_mask: Bool[Tensor, "seq_len seq_len"] = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        mask = base_mask[None, None, :, :]  # (1,1,seq,seq) -> broadcast
        # apply scaled dot product attention for each head
        attn_output_heads: Float[Tensor, "... head seq_len d_v"] = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        # reshape back to (..., seq_len, d_model)
        attn_output = rearrange(attn_output_heads, "... head seq_len d_v -> ... seq_len (head d_v)", head=self.num_heads)
        # final linear projection
        output: Float[Tensor, "... seq_len d_in"] = self.w_o(attn_output)
        return output




