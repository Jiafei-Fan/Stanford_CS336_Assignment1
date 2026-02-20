import torch
import torch.nn as nn
from jaxtyping import Float, Bool, Int
from torch import Tensor
from einops import einsum, rearrange
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear
from .softmax import softmax

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int, 
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta)
            for _ in range(num_layers)
        ])
        self.final_rmsnorm = RMSNorm(d_model)
        self.output_projection = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, token_ids: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        # Input embedding
        x: Float[Tensor, "batch_size sequence_length d_model"] = self.token_embeddings(token_ids)
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        # Final RMSNorm
        x = self.final_rmsnorm(x)
        # Output projection to vocab size
        logits: Float[Tensor, "batch_size sequence_length vocab_size"] = self.output_projection(x)
        return logits