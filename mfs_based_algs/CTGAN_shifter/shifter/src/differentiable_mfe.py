"""
Differentiable meta-features for experiment_iris (only the MFs in use).

MF list: gravity, w_lambda, p_trace, lh_trace, roy_root, sd_ratio,
         mean, sd, var, max, min, range, h_mean,
         eigenvalues, cor, cov,
         mad, t_mean, sparsity,
         can_cor,
         attr_ent, joint_ent.
"""

from __future__ import annotations

import numpy as np
import torch


# ============== STATISTICAL (differentiable) ==============

def ft_mean(N: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Mean per feature. (n, d) -> (d,)."""
    return N.mean(dim=dim)


def ft_var(N: torch.Tensor, ddof: int = 1, dim: int = 0) -> torch.Tensor:
    """Variance per feature."""
    n = N.shape[dim]
    return N.var(dim=dim, unbiased=(ddof == 1))


def ft_sd(N: torch.Tensor, ddof: int = 1, dim: int = 0) -> torch.Tensor:
    """Standard deviation."""
    return ft_var(N, ddof=ddof, dim=dim).sqrt()


def ft_max(N: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Max per feature via torch.quantile(1.0) (supports backward in PyTorch)."""
    return torch.quantile(N, 1.0, dim=dim, interpolation="linear")


def ft_min(N: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Min per feature via torch.quantile(0.0) (supports backward in PyTorch)."""
    return torch.quantile(N, 0.0, dim=dim, interpolation="linear")


def ft_range(N: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Range (max - min) via torch.quantile: Q(1) - Q(0) (supports backward in PyTorch)."""
    return torch.quantile(N, 1.0, dim=dim, interpolation="linear") - torch.quantile(
        N, 0.0, dim=dim, interpolation="linear"
    )


def ft_cov(N: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    """Lower triangle of the covariance matrix (excluding the diagonal)."""
    n, d = N.shape
    c = (N - N.mean(0)).T @ (N - N.mean(0)) / (n - ddof)
    idx = torch.tril_indices(d, d, offset=-1)
    return torch.abs(c[idx[0], idx[1]])


def ft_cor(N: torch.Tensor) -> torch.Tensor:
    """Absolute correlations (lower triangle)."""
    c = torch.corrcoef(N.T).abs()
    d = N.shape[1]
    idx = torch.tril_indices(d, d, offset=-1)
    return c[idx[0], idx[1]]


def ft_eigenvalues(N: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    """Eigenvalues of the covariance matrix (ascending order)."""
    n, d = N.shape
    cov = (N - N.mean(0)).T @ (N - N.mean(0)) / (n - ddof)
    eigs = torch.linalg.eigvalsh(cov)
    return eigs


def ft_h_mean(N: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Harmonic mean: n / sum(1/x)."""
    safe = torch.clamp(N, min=epsilon)
    inv_sum = (1.0 / safe).sum(0)
    n = N.shape[0]
    return torch.tensor(n, dtype=N.dtype, device=N.device) / inv_sum


# --------------- Additional statistical ---------------

def ft_mad(N: torch.Tensor, factor: float = 1.4826, dim: int = 0) -> torch.Tensor:
    """Median Absolute Deviation. As in pymfe/scipy: raw_MAD * factor (scale=1/factor)."""
    med = torch.quantile(N, 0.5, dim=dim, interpolation="linear")
    dev = (N - med.unsqueeze(0)).abs()
    mad_val = torch.quantile(dev, 0.5, dim=dim, interpolation="linear")
    return mad_val * factor


def ft_t_mean(N: torch.Tensor, pcut: float = 0.2, dim: int = 0) -> torch.Tensor:
    """Trimmed mean: drop pcut fraction from each tail, then take the mean."""
    n = N.shape[dim]
    k = int(n * pcut)
    if k < 1:
        return N.mean(dim=dim)
    sorted_vals, _ = torch.sort(N, dim=dim)
    trimmed = sorted_vals.narrow(dim, k, n - 2 * k)
    return trimmed.mean(dim=dim)


# --------------- Canonical correlation: PyTorch (differentiable) ---------------

def _inv_sqrt_matrix(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """A^{-1/2} via eigendecomposition. A must be symmetric."""
    L, V = torch.linalg.eigh(A)
    L = L.clamp(min=eps)
    return V @ (1.0 / L.sqrt()).diag() @ V.T


def ft_can_cor(
    N: torch.Tensor,
    y_onehot: torch.Tensor,
    ddof: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Canonical correlations between N and y_onehot (drop-first column). Differentiable in N.
    C = Sigma_xx^{-1/2} Sigma_xy Sigma_yy^{-1/2}; can_cors are singular values of C."""
    n, dx = N.shape
    _, dy = y_onehot.shape
    n_comp = min(dx, dy)
    if n_comp < 1:
        return torch.tensor([], device=N.device, dtype=N.dtype)
    X = N - N.mean(0)
    Y = y_onehot - y_onehot.mean(0)
    Sxx = (X.T @ X) / (n - ddof) + eps * torch.eye(dx, device=N.device, dtype=N.dtype)
    Syy = (Y.T @ Y) / (n - ddof) + eps * torch.eye(dy, device=N.device, dtype=N.dtype)
    Sxy = (X.T @ Y) / (n - ddof)
    Sxx_inv_sqrt = _inv_sqrt_matrix(Sxx, eps=eps)
    Syy_inv_sqrt = _inv_sqrt_matrix(Syy, eps=eps)
    C = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
    U, S, _ = torch.linalg.svd(C)
    can_cors = S[:n_comp].clamp(0, 1)
    return can_cors


def _can_cor_to_eigval_torch(can_cors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """eig_i = can_cor_i^2 / (1 - can_cor_i^2)."""
    sq = can_cors.square()
    return sq / (1.0 - sq.clamp(max=1 - eps))


def ft_w_lambda(N: torch.Tensor, y_onehot: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    """Wilks' Lambda: prod(1/(1+can_cor_eigval_i)). Differentiable."""
    cc = ft_can_cor(N, y_onehot, ddof=ddof)
    if cc.numel() == 0:
        return torch.tensor(float('nan'), device=N.device, dtype=N.dtype)
    eigvals = _can_cor_to_eigval_torch(cc)
    return torch.exp(-torch.log1p(eigvals).sum()).unsqueeze(0)


def ft_p_trace(N: torch.Tensor, y_onehot: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    """Pillai's trace: sum(can_cor_i^2). Differentiable."""
    cc = ft_can_cor(N, y_onehot, ddof=ddof)
    if cc.numel() == 0:
        return torch.tensor(float('nan'), device=N.device, dtype=N.dtype)
    return cc.square().sum().unsqueeze(0)


def ft_lh_trace(N: torch.Tensor, y_onehot: torch.Tensor, ddof: int = 1) -> torch.Tensor:
    """Lawley-Hotelling trace: sum(can_cor_eigval_i). Differentiable."""
    cc = ft_can_cor(N, y_onehot, ddof=ddof)
    if cc.numel() == 0:
        return torch.tensor(float('nan'), device=N.device, dtype=N.dtype)
    eigvals = _can_cor_to_eigval_torch(cc)
    return eigvals.sum().unsqueeze(0)


def ft_roy_root(N: torch.Tensor, y_onehot: torch.Tensor, ddof: int = 1, criterion: str = "eigval") -> torch.Tensor:
    """Roy's largest root. Differentiable."""
    cc = ft_can_cor(N, y_onehot, ddof=ddof)
    if cc.numel() == 0:
        return torch.tensor(float('nan'), device=N.device, dtype=N.dtype)
    if criterion == "eigval":
        eigvals = _can_cor_to_eigval_torch(cc)
        return eigvals.max().unsqueeze(0)
    return cc.square().max().unsqueeze(0)


def ft_gravity(N: torch.Tensor, y_onehot: torch.Tensor, norm_ord: float = 2.0) -> torch.Tensor:
    """Distance between class-weighted centers of majority vs. minority. Differentiable in N."""
    n, d = N.shape
    n_classes = y_onehot.shape[1]
    counts = y_onehot.sum(0)
    ind_maj = counts.argmax()
    mask_min = torch.ones(n_classes, dtype=torch.bool, device=N.device)
    mask_min[ind_maj] = False
    counts_rest = counts.masked_fill(~mask_min, float('inf'))
    ind_min = counts_rest.argmin()
    w_maj = y_onehot[:, ind_maj:ind_maj + 1]
    w_min = y_onehot[:, ind_min:ind_min + 1]
    n_maj = w_maj.sum().clamp(min=1e-8)
    n_min = w_min.sum().clamp(min=1e-8)
    center_maj = (N * w_maj).sum(0) / n_maj
    center_min = (N * w_min).sum(0) / n_min
    return torch.linalg.norm(center_maj - center_min, ord=norm_ord).unsqueeze(0)


def ft_sparsity(N: torch.Tensor, num_bins: int = 50, normalize: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """Sparsity via soft bins: effective_count = 1/sum(p^2), S = (n/ec - 1)/(n-1). Differentiable."""
    n, d = N.shape
    B = min(num_bins, max(2, n // 2))
    out = []
    for j in range(d):
        w = _soft_bin_weights_1d(N[:, j], B, sigma_scale=0.5, eps=eps)
        p = w.mean(0)
        ec = 1.0 / (p.square().sum() + eps)
        s = (n / ec - 1.0)
        if normalize and n > 1:
            s = s / (n - 1)
        out.append(s)
    return torch.stack(out)


def ft_sd_ratio(N: torch.Tensor, y_onehot: torch.Tensor, ddof: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Homogeneity of covariances (Box M-style statistic). Differentiable in N."""
    n, d = N.shape
    n_classes = y_onehot.shape[1]
    if n_classes < 2:
        return torch.tensor(float('nan'), device=N.device, dtype=N.dtype).unsqueeze(0)
    sample_covs = []
    vec_weight = []
    for c in range(n_classes):
        w = y_onehot[:, c:c + 1]
        nc = w.sum().clamp(min=1)
        mean_c = (N * w).sum(0) / nc
        X_centered = (N - mean_c) * w
        Sc = (X_centered.T @ X_centered) / (nc - ddof) + eps * torch.eye(d, device=N.device, dtype=N.dtype)
        sample_covs.append(Sc)
        vec_weight.append((nc - 1).clamp(min=1e-8))
    vec_weight = torch.stack(vec_weight)
    pooled = sum(Sc * w for Sc, w in zip(sample_covs, vec_weight)) / (n - n_classes)
    log_det_pooled = torch.linalg.slogdet(pooled)[1]
    log_dets = torch.stack([torch.linalg.slogdet(Sc)[1] for Sc in sample_covs])
    gamma = 1.0 - ((2 * d**2 + 3 * d - 1) / (6 * (d + 1) * (n_classes - 1))) * (
        (1.0 / vec_weight).sum() - 1.0 / (n - n_classes)
    )
    m_factor = gamma * ((n - n_classes) * log_det_pooled - (vec_weight * log_dets).sum())
    if torch.isinf(m_factor):
        return torch.tensor(float('nan'), device=N.device, dtype=N.dtype).unsqueeze(0)
    return torch.exp(m_factor / (d * (n - n_classes))).unsqueeze(0)


# ============== INFO-THEORY (soft bins, differentiable in N) ==============
# Bin edges: torch.quantile (backward supported). Assignment: soft (Gaussian kernel),
# then H = -sum(p*log(p)) — gradients flow through the data.


def _soft_bin_weights_1d(
    x: torch.Tensor,
    num_bins: int,
    sigma_scale: float = 0.5,
    eps: float = 1e-8,
    boundaries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft membership weights over bins for vector x. (n,) -> (n, num_bins).
    Bin edges from quantiles of x (or fixed ``boundaries`` if provided)."""
    if boundaries is None:
        q_levels = torch.linspace(0.0, 1.0, num_bins + 1, device=x.device, dtype=x.dtype)
        boundaries = torch.quantile(x, q_levels, dim=0, interpolation="linear")
    if boundaries.numel() < 2:
        one_bin = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return one_bin
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    width = (boundaries[1:] - boundaries[:-1]).clamp(min=eps)
    sigma = (width * sigma_scale).clamp(min=eps)
    dist = (x.unsqueeze(1) - centers.unsqueeze(0)) ** 2
    sigma_sq = sigma.unsqueeze(0).clamp(min=1e-10)
    log_w = -dist / (2 * sigma_sq)
    w = torch.softmax(log_w, dim=1)
    return w


def _entropy_soft(p: torch.Tensor, eps: float = 1e-8, base: float = 2.0) -> torch.Tensor:
    """Shannon entropy of distribution p (last axis = categories). H = -sum(p*log(p))."""
    p_safe = (p + eps).clamp(max=1.0)
    return -(p_safe * torch.log(p_safe)).sum(dim=-1) / (torch.log(torch.tensor(base, device=p.device, dtype=p.dtype)))


def get_attr_ent_boundaries(N: torch.Tensor, num_bins: int | None = None) -> torch.Tensor:
    """Per-feature bin edges (d, B+1). Use for fixed bins during training."""
    n, d = N.shape
    B = num_bins if num_bins is not None else max(2, int(n ** (1 / 3)))
    q_levels = torch.linspace(0.0, 1.0, B + 1, device=N.device, dtype=N.dtype)
    boundaries = torch.stack([
        torch.quantile(N[:, j], q_levels, dim=0, interpolation="linear") for j in range(d)
    ])
    return boundaries


def ft_attr_ent(
    N: torch.Tensor,
    num_bins: int | None = None,
    sigma_scale: float = 0.5,
    eps: float = 1e-8,
    boundaries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-feature entropy with soft bins. If ``boundaries`` (d, B+1) is set, edges are fixed."""
    n, d = N.shape
    B = num_bins if num_bins is not None else max(2, int(n ** (1 / 3)))
    ent = []
    for j in range(d):
        b_j = boundaries[j] if boundaries is not None else None
        w = _soft_bin_weights_1d(N[:, j], B, sigma_scale=sigma_scale, eps=eps, boundaries=b_j)
        p = w.mean(0)
        ent.append(_entropy_soft(p.unsqueeze(0), eps=eps).squeeze(0))
    return torch.stack(ent)


def ft_joint_ent(
    N: torch.Tensor,
    y_onehot: torch.Tensor,
    num_bins: int | None = None,
    sigma_scale: float = 0.5,
    eps: float = 1e-8,
    boundaries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Joint entropy H(x_j, y). If ``boundaries`` (d, B+1) is set, x uses fixed bin edges."""
    n, d = N.shape
    B = num_bins if num_bins is not None else max(2, int(n ** (1 / 3)))
    ent = []
    for j in range(d):
        b_j = boundaries[j] if boundaries is not None else None
        w = _soft_bin_weights_1d(N[:, j], B, sigma_scale=sigma_scale, eps=eps, boundaries=b_j)
        p_joint = (w.unsqueeze(2) * y_onehot.unsqueeze(1)).sum(0) / n
        ent.append(_entropy_soft(p_joint.reshape(1, -1), eps=eps).squeeze(0))
    return torch.stack(ent)


# ============== Aggregation and extraction ==============

# MFs used in experiment_iris (META_FEATURES_SINGLE)
ALL_FEATURES = [
    "gravity", "w_lambda", "p_trace", "lh_trace", "roy_root", "sd_ratio",
    "mean", "sd", "var", "max", "min", "range",
    "h_mean", "skewness", "kurtosis", "iq_range", "median",
    "eigenvalues", "cor", "cov",
    "mad", "t_mean", "sparsity",
    "can_cor",
    "attr_ent", "joint_ent",
]


def extract_torch_mfe(
    N: torch.Tensor,
    features: list[str] | None = None,
    ddof: int = 1,
    y: torch.Tensor | np.ndarray | None = None,
    cat_cols: list[int] | None = None,
    attr_ent_boundaries: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    Extract meta-features from numeric matrix N (n, d).
    ``y`` is the target label. ``attr_ent_boundaries`` optional (d, B+1) for fixed attr_ent/joint_ent bins.
    """
    if features is None:
        features = list(ALL_FEATURES)

    y_onehot = None
    y_onehot_full = None
    if y is not None:
        y_np = np.asarray(y).ravel()
        n_classes = int(y_np.max()) + 1
        y_onehot_full = torch.nn.functional.one_hot(
            torch.tensor(y_np, dtype=torch.long, device=N.device), n_classes
        ).to(N.dtype)
        y_onehot = y_onehot_full[:, 1:]  # drop first for CCA

    results = {}
    names = []

    for f in features:
        if f == "mean":
            v = ft_mean(N)
        elif f == "sd":
            v = ft_sd(N, ddof=ddof)
        elif f == "var":
            v = ft_var(N, ddof=ddof)
        elif f == "max":
            v = ft_max(N)
        elif f == "min":
            v = ft_min(N)
        elif f == "range":
            v = ft_range(N)
        elif f == "cor":
            v = ft_cor(N)
        elif f == "cov":
            v = ft_cov(N, ddof=ddof)
        elif f == "eigenvalues":
            v = ft_eigenvalues(N, ddof=ddof)
        elif f == "h_mean":
            v = ft_h_mean(N)
        elif f == "mad":
            v = ft_mad(N)
        elif f == "t_mean":
            v = ft_t_mean(N)
        elif f == "sparsity":
            v = ft_sparsity(N)
        elif f == "gravity" and y_onehot_full is not None:
            v = ft_gravity(N, y_onehot_full)
        elif f == "can_cor" and y_onehot is not None:
            v = ft_can_cor(N, y_onehot, ddof=ddof)
        elif f == "w_lambda" and y_onehot is not None:
            v = ft_w_lambda(N, y_onehot, ddof=ddof)
        elif f == "p_trace" and y_onehot is not None:
            v = ft_p_trace(N, y_onehot, ddof=ddof)
        elif f == "lh_trace" and y_onehot is not None:
            v = ft_lh_trace(N, y_onehot, ddof=ddof)
        elif f == "roy_root" and y_onehot is not None:
            v = ft_roy_root(N, y_onehot, ddof=ddof)
        elif f == "sd_ratio" and y_onehot_full is not None:
            v = ft_sd_ratio(N, y_onehot_full, ddof=ddof)
        elif f == "attr_ent":
            v = ft_attr_ent(N, boundaries=attr_ent_boundaries)
        elif f == "joint_ent" and y_onehot_full is not None:
            v = ft_joint_ent(N, y_onehot_full, boundaries=attr_ent_boundaries)
        else:
            continue
        results[f] = v
        names.append(f)

    return results, names


def flatten_results(results: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[str]]:
    """Flatten results to one vector and names with suffixes for vector-valued MFs."""
    values = []
    names = []
    for k, v in results.items():
        v_flat = v.flatten()
        for i in range(v_flat.shape[0]):
            values.append(v_flat[i])
            names.append(f"{k}_{i}" if v_flat.shape[0] > 1 else k)
    if values:
        return torch.stack(values), names
    return torch.tensor([], dtype=torch.float32), []
