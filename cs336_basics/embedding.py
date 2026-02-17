import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int, Int64
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ) -> None:
        """
        num_embeddings: size of the dictionary of embeddings
        embedding_dim: the size of each embedding vector
        device: the device of the parameters
        dtype: the datatype of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings # vocab_size
        self.embedding_dim = embedding_dim # d_model
        # store
        self.weight: Float[Tensor, "vocab_size d_model"] = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        std: float = 1.0 / (embedding_dim ** 0.5)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: Int64[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        """
        Lookup the embedding vectors for the given
        """
        # output shape: (batch_size, sequence_length, d_model)
        return self.weight[token_ids]