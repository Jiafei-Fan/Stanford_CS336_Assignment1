import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from einops import einsum
from .linear import Linear

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
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def silu(self, x: Float[Tensor, "... d_ff"]) -> Float[Tensor, "... d_ff"]:
        return x * torch.sigmoid(x)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        x: input tensor of shape (..., d_model)
        returns: output tensor of shape (..., d_model) after applying the position-wise feedforward network, with 3 weight matrices
        """
        first_linear: Float[Tensor, "... d_ff"] = self.w1(x)
        silu = self.silu(first_linear)
        second_linear: Float[Tensor, "... d_ff"] = self.w3(x)
        elemetwise_product: Float[Tensor, "... d_ff"] = einsum(silu, second_linear, "... d_ff, ... d_ff -> ... d_ff")
        result: Float[Tensor, "... d_model"] = self.w2(elemetwise_product)
        return result

