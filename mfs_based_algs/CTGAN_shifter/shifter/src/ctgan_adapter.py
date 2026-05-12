"""Adapter wrapping a trained CTGAN model for noise-based generation."""

from __future__ import annotations

import os
import sys
from typing import Union

import numpy as np
import pandas as pd
import torch


class CTGANRepoAdapter:
    """Wrapper around a trained CTGAN for generation via direct noise injection."""

    def __init__(self, ctgan_model):
        self.m = ctgan_model

    @staticmethod
    def load(path: str, device: str = "cpu", ctgan_lib_parent: str | None = None) -> CTGANRepoAdapter:
        """Load a CTGAN .pkl checkpoint and return an adapter."""
        if ctgan_lib_parent is None:
            ctgan_lib_parent = os.path.dirname(os.path.abspath(path))
        ctgan_lib_parent = os.path.abspath(ctgan_lib_parent)
        if ctgan_lib_parent not in sys.path:
            sys.path.insert(0, ctgan_lib_parent)

        model = torch.load(path, map_location=device, weights_only=False)
        if hasattr(model, "set_device"):
            model.set_device(device)
        return CTGANRepoAdapter(model)

    @property
    def z_dim(self) -> int:
        return int(self.m._embedding_dim)

    @property
    def device(self) -> torch.device:
        return next(self.m._generator.parameters()).device

    def generate_from_noise(
        self,
        Z: Union[torch.Tensor, np.ndarray],
        cond_vec: np.ndarray | None = None,
        batch_rows: int = 4096,
    ) -> pd.DataFrame:
        """Run noise Z through the CTGAN generator → DataFrame.

        Parameters
        ----------
        Z        : [n_samples, z_dim]
        cond_vec : pre-sampled conditional vectors (if None, sampled randomly)
        """
        if isinstance(Z, torch.Tensor):
            Z = Z.detach().cpu().numpy()
        Z = np.asarray(Z, dtype=np.float32)
        assert Z.shape[1] == self.z_dim, f"Expected z_dim={self.z_dim}, got {Z.shape[1]}"

        device = self.device
        all_data = []

        for start in range(0, Z.shape[0], batch_rows):
            end = min(start + batch_rows, Z.shape[0])
            z_batch = torch.tensor(Z[start:end], dtype=torch.float32, device=device)

            if cond_vec is not None:
                cv_batch = cond_vec[start:end]
            else:
                cv_batch = self.m._data_sampler.sample_original_condvec(z_batch.shape[0])

            if cv_batch is not None:
                c = torch.from_numpy(cv_batch).to(device)
                z_input = torch.cat([z_batch, c], dim=1)
            else:
                z_input = z_batch

            with torch.no_grad():
                fake = self.m._generator(z_input)
                fakeact = self.m._apply_activate(fake)
            all_data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(all_data, axis=0)
        result = self.m._transformer.inverse_transform(data)
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    def sample_cond_vec(self, n: int) -> np.ndarray | None:
        """Pre-sample conditional vectors (for deterministic differentiable runs)."""
        return self.m._data_sampler.sample_original_condvec(n)

    def generate_from_noise_differentiable(
        self,
        Z: torch.Tensor,
        cond_vec: np.ndarray | None = None,
        batch_rows: int = 4096,
    ) -> torch.Tensor:
        """Differentiable generation: Z → Tensor (preserves computation graph).

        Applies the CTGAN generator without detaching, then performs a
        differentiable inverse-transform for continuous columns using
        VGM parameters and a straight-through estimator for component selection.

        Parameters
        ----------
        Z        : [n_samples, z_dim]
        cond_vec : pre-sampled conditional vectors (if None, sampled randomly)
        """
        assert Z.shape[1] == self.z_dim, f"Expected z_dim={self.z_dim}, got {Z.shape[1]}"
        device = self.device
        all_data = []

        for start in range(0, Z.shape[0], batch_rows):
            end = min(start + batch_rows, Z.shape[0])
            z_batch = Z[start:end].to(device)

            if cond_vec is not None:
                cv_batch = cond_vec[start:end]
            else:
                cv_batch = self.m._data_sampler.sample_original_condvec(z_batch.shape[0])

            if cv_batch is not None:
                c = torch.from_numpy(cv_batch).to(device).requires_grad_(False)
                z_input = torch.cat([z_batch, c], dim=1)
            else:
                z_input = z_batch

            fake = self.m._generator(z_input)
            fakeact = self.m._apply_activate(fake)
            all_data.append(fakeact)

        data = torch.cat(all_data, dim=0)

        # Differentiable inverse-transform per column
        transformed_data = []
        st = 0

        for col_info in self.m._transformer._column_transform_info_list:
            dim = col_info.output_dimensions
            column_data = data[:, st:st + dim]

            if col_info.column_type == "continuous":
                u = column_data[:, 0]  # tanh-activated value in [-1, 1]
                v = column_data[:, 1:]  # mixture component logits

                gm = col_info.transform
                bgm = gm._bgm_transformer
                valid = gm.valid_component_indicator
                means = torch.tensor(bgm.means_.reshape(-1)[valid], dtype=torch.float32, device=device)
                stds = torch.tensor(np.sqrt(bgm.covariances_.reshape(-1)[valid]), dtype=torch.float32, device=device)

                # Straight-through estimator: argmax forward, softmax backward
                v_soft = torch.softmax(v, dim=1)
                v_hard_idx = torch.argmax(v, dim=1)

                mean_hard = means[v_hard_idx]
                std_hard = stds[v_hard_idx]
                mean_soft = (v_soft @ means.unsqueeze(1)).squeeze(1)
                std_soft = (v_soft @ stds.unsqueeze(1)).squeeze(1)

                mean_st = mean_hard + mean_soft - mean_soft.detach()
                std_st = std_hard + std_soft - std_soft.detach()

                recovered = u * 4.0 * std_st + mean_st
                transformed_data.append(recovered.unsqueeze(1))
            else:
                # Categorical: argmax (non-differentiable by nature)
                recovered = torch.argmax(column_data, dim=1).float().unsqueeze(1).detach()
                transformed_data.append(recovered)

            st += dim

        return torch.cat(transformed_data, dim=1)
