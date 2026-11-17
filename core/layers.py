import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NovaConfig


class PlasticLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 0.0

        if rank > 0:
            self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(rank)))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.plastic_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.plastic_enabled and self.rank > 0:
            plastic_delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
            out = out + plastic_delta
        return out


class AdaRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.persona_proj = nn.Linear(dim, dim * 2, bias=True)

        # Initialize modulation to zero so base normalization starts neutral
        nn.init.zeros_(self.persona_proj.weight)
        nn.init.zeros_(self.persona_proj.bias)

    def forward(self, x: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        gamma, beta = self.persona_proj(z_p).chunk(2, dim=-1)
        return normed * (self.weight + gamma) + beta


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0, device: str = "cpu") -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[1]].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class NovaAttention(nn.Module):
    def __init__(self, cfg: NovaConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.wq = PlasticLinear(cfg.d_model, cfg.d_model, cfg.lora_rank, cfg.lora_alpha)
        self.wk = PlasticLinear(cfg.d_model, cfg.d_model, cfg.lora_rank, cfg.lora_alpha)
        self.wv = PlasticLinear(cfg.d_model, cfg.d_model, cfg.lora_rank, cfg.lora_alpha)
        self.wo = PlasticLinear(cfg.d_model, cfg.d_model, cfg.lora_rank, cfg.lora_alpha)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class NovaSwiGLU(nn.Module):
    def __init__(self, cfg: NovaConfig):
        super().__init__()
        self.w1 = PlasticLinear(cfg.d_model, cfg.d_ff, cfg.lora_rank, cfg.lora_alpha)
        self.w2 = PlasticLinear(cfg.d_ff, cfg.d_model, cfg.lora_rank, cfg.lora_alpha)
        self.w3 = PlasticLinear(cfg.d_model, cfg.d_ff, cfg.lora_rank, cfg.lora_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class NovaBlock(nn.Module):
    def __init__(self, cfg: NovaConfig):
        super().__init__()
        self.attn_norm = AdaRMSNorm(cfg.d_model)
        self.attn = NovaAttention(cfg)
        self.ffn_norm = AdaRMSNorm(cfg.d_model)
        self.ffn = NovaSwiGLU(cfg)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x, z_p), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x, z_p))
        return x