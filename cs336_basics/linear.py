import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
                 ) -> None:
        """
        in_features: size of each input sample
        out_features: size of each output sample
        device: the device of the parameters
        dtype: the datatype of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # store
        self.weight: Float[Tensor, "d_in d_out"] = nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        std: float = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    
    def forward(self, x: Float[Tensor, "... d_in"]) -> torch.Tensor:
        """
        Apply the linear transformation to the input
        """
        result = einsum(x, self.weight, "... in_features, in_features out_features -> ... out_features")
        return result

