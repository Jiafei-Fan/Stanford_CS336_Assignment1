import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None
    ) -> None:
        """
        d_model: hidden dimension of the input
        eps: Epsilon value for numberical stability
        device: Device to store the parameters on
        dtype: Datatype of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight: Float[Tensor, "d_model"] = nn.Parameter(torch.empty((d_model), device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "batch_size sequence_length d_model"]) -> Float[Tensor, "batch_size sequence_length d_model"]:
        """
        x: input tensor of shape (batch_size, sequence_length, d_model)
        returns: experience normalization, output tensor of shape (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x_f = x.to(torch.float32)
        mean_square = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_normed: Float[Tensor, "batch_size sequence_length d_model"] = x_f / torch.sqrt(mean_square + self.eps)
        result = einsum(x_normed, self.weight, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)
        