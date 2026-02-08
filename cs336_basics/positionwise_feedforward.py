import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from einops import einsum

class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None
    ) -> None:
        """
        d_model: hidden dimension of the input
        d_ff: hidden dimension of the feedforward network
        device: Device to store the parameters on
        dtype: Datatype of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1: Float[Tensor, "d_ff d_model"] = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2: Float[Tensor, "d_model d_ff"] = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3: Float[Tensor, "d_ff d_model"] = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "batch_size sequence_length d_model"]:
        first_linear = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        sigmoid = first_linear * torch.sigmoid(first_linear)
        second_linear = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        elemetwise_product = einsum(sigmoid, second_linear, "... d_ff, ... d_ff -> ... d_ff")
        result = einsum(elemetwise_product, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        return result

