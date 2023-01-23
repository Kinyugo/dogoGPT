import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchtyping import TensorType

HiddenFeaturesType = TensorType["batch", "seq_len", "d_model"]


class CausalConv1d(nn.Module):
    def __init__(
        self, d_in: int, d_out: int, kernel_size: int = 5, dilation: int = 1
    ) -> None:
        super().__init__()

        self.padding = (kernel_size - 1) * dilation
        self.fn = nn.Conv1d(
            d_in,
            d_out,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(
        self, x: TensorType["batch", "seq_len", "d_in"]
    ) -> TensorType["batch", "seq_len", "d_out"]:
        # Switch to channels first representation
        x = rearrange(x, "b l c -> b c l")

        # Apply the convolution and remove the padding at the end
        x = self.fn(x)
        if self.padding != 0:
            x = x[..., : -self.padding]

        # Switch to channels last representation
        x = rearrange(x, "b c l -> b l c")

        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff_factor: int = 4,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.fn = nn.Sequential(
            nn.LayerNorm(d_model),
            CausalConv1d(d_model, d_model, kernel_size, dilation),
            nn.GELU(),
            nn.Linear(d_model, d_model * d_ff_factor),
            nn.GELU(),
            nn.Linear(d_model * d_ff_factor, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: HiddenFeaturesType) -> HiddenFeaturesType:
        return self.fn(x) + x


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: HiddenFeaturesType) -> HiddenFeaturesType:
        return x + self._attn_forward(x)

    def _attn_forward(self, x: HiddenFeaturesType) -> HiddenFeaturesType:
        x = self.norm(x)
        x, _ = self.multihead_attn(
            x, x, x, attn_mask=self._generate_attn_mask(x.shape[-2], x.device)
        )

        return x

    def _generate_attn_mask(
        self, seq_len: int, device: torch.device
    ) -> TensorType["seq_len", "seq_len"]:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff_factor: int = 4,
        kernel_size: int = 5,
        dilation: int = 1,
        n_heads: int = 8,
        dropout: float = 0.0,
        has_attention: bool = False,
    ) -> None:
        super().__init__()

        self.residual_block = ResidualBlock(
            d_model, d_ff_factor, kernel_size, dilation, dropout
        )
        self.attention = (
            Attention(d_model, n_heads, dropout) if has_attention else nn.Identity()
        )

    def forward(self, x: HiddenFeaturesType) -> HiddenFeaturesType:
        return self.attention(self.residual_block(x))
