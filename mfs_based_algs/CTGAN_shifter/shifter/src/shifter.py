"""Shifter network: per-sample latent-space manipulation with DeepSets pooling.

Architecture:
    c       = MetaEncoder(m*)                       # target meta-features → conditioning
    mu_Z    = mean(Z, dim=samples)                  # permutation-invariant pooling
    delta_i = MLP(z_i || c || mu_Z)                 # per-sample shift
    z_tilde_i = z_i + delta_scale * delta_i         # residual connection
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MetaEncoder(nn.Module):
    """MLP: target meta-feature vector m* → conditioning vector c."""

    def __init__(self, m_dim: int, c_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, c_dim),
        )

    def forward(self, m_star: torch.Tensor) -> torch.Tensor:
        return self.net(m_star)  # [B, m_dim] → [B, c_dim]


class Shifter(nn.Module):
    """Per-sample noise shifter with DeepSets pooling (mean or max)."""

    def __init__(self, z_dim: int, m_dim: int, c_dim: int = 64,
                 hidden_dim: int = 256, delta_scale: float = 0.1,
                 pooling: str = "mean"):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.delta_scale = delta_scale
        self.pooling = pooling.lower()
        if self.pooling not in ["mean", "max"]:
            raise ValueError(f"pooling must be 'mean' or 'max', got {pooling}")
        
        self.encoder = MetaEncoder(m_dim, c_dim)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + c_dim + z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, Z_base: torch.Tensor, m_star: torch.Tensor) -> torch.Tensor:
        """Z_base: [B, N, z_dim], m_star: [B, m_dim] → Z_tilde: [B, N, z_dim]"""
        B, N, z_dim = Z_base.shape

        c = self.encoder(m_star)                                      # [B, c_dim]
        c_exp = c.unsqueeze(1).expand(B, N, self.c_dim)               # [B, N, c_dim]
        
        # DeepSets pooling: mean or max
        if self.pooling == "mean":
            pool_Z = Z_base.mean(dim=1, keepdim=True).expand(B, N, z_dim)
        else:  # max
            pool_Z = Z_base.max(dim=1, keepdim=True)[0].expand(B, N, z_dim)

        inp = torch.cat([Z_base, c_exp, pool_Z], dim=-1)              # [B, N, input_dim]
        delta = self.mlp(inp.reshape(B * N, -1)).reshape(B, N, z_dim)
        
        return Z_base + self.delta_scale * delta


def latent_reg(Z_tilde: torch.Tensor, Z_base: torch.Tensor) -> torch.Tensor:
    """L_z = mean(||z_tilde - z_base||^2) — keeps shifted noise close to the prior."""
    return ((Z_tilde - Z_base) ** 2).mean()


def feature_space_reg(X_tilde: torch.Tensor, X_base: torch.Tensor) -> torch.Tensor:
    """L_x = mean((X_tilde - X_base)^2) — feature-space proximity."""
    return ((X_tilde - X_base) ** 2).mean()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.delta_scale = delta_scale
        self.encoder = MetaEncoder(m_dim, c_dim)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + c_dim + z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, Z_base: torch.Tensor, m_star: torch.Tensor) -> torch.Tensor:
        """Z_base: [B, N, z_dim], m_star: [B, m_dim] → Z_tilde: [B, N, z_dim]"""
        B, N, z_dim = Z_base.shape

        c = self.encoder(m_star)                                      # [B, c_dim]
        c_exp = c.unsqueeze(1).expand(B, N, self.c_dim)               # [B, N, c_dim]
        mu_Z = Z_base.mean(dim=1, keepdim=True).expand(B, N, z_dim)   # DeepSets pooling

        inp = torch.cat([Z_base, c_exp, mu_Z], dim=-1)                # [B, N, input_dim]
        delta = self.mlp(inp.reshape(B * N, -1)).reshape(B, N, z_dim)

        return Z_base + self.delta_scale * delta


def latent_reg(Z_tilde: torch.Tensor, Z_base: torch.Tensor) -> torch.Tensor:
    """L_z = mean(||z_tilde - z_base||^2) — keeps shifted noise close to the prior."""
    return ((Z_tilde - Z_base) ** 2).mean()


def feature_space_reg(X_tilde: torch.Tensor, X_base: torch.Tensor) -> torch.Tensor:
    """L_x = mean((X_tilde - X_base)^2) — feature-space proximity."""
    return ((X_tilde - X_base) ** 2).mean()
