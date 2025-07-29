import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

class AdversarialLabelDRO(nn.Module):
    """Distributionally Robust Optimization model for adversarial label shift.
    Implements the method from "Coping with label shift via distributionally robust optimization" (Zhang et al., ICLR 2021).
    Designed for binary classification with tabular data.
    
    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        n_groups: Number of classes (must be 2 for binary classification)
        eta_pi: Learning rate for dual variable
        r: Radius for KL constraint
        clip_max: Gradient clipping value (default: 2.0)
        eps: Smoothing factor for pi (default: 0.001)
        beta: EMA decay factor for empirical distribution (default: 0.999)
        lr: Learning rate for model optimizer (default: 0.01)
        activation: Activation function ('relu' or 'tanh', default: 'relu')
        dropout: Dropout probability (default: 0.0)
        weight_decay: L2 regularization (default: 0.0)
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [64, 32],
                 n_groups: int = 2,
                 eta_pi: float = 0.1,
                 r: float = 0.1,
                 clip_max: float = 2.0,
                 eps: float = 0.001,
                 beta: float = 0.999,
                 lr: float = 0.01,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 weight_decay: float = 0.0):
        super().__init__()
        self.n_groups = n_groups
        self.eta_pi = eta_pi
        self.r = r
        self.clip_max = clip_max
        self.eps = eps
        self.beta = beta
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        
        # Initialize MLP layers
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize distributions
        self.register_buffer('p_emp', torch.full((n_groups,), 1./n_groups))
        self.pi_t = nn.Parameter(torch.full((n_groups,), 1./n_groups), requires_grad=True)
        
        # Optimizer (only for model parameters)
        self.optimizer = torch.optim.Adam(
            self.mlp.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 10,
            batch_size: int = 32,
            device: str = "cpu") -> None:
        """Train the model on tabular data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Binary labels (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device for training ('cpu' or 'cuda')
        """
        # Validate input dimensions
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Model initialized with input_dim={self.input_dim}, "
                             f"but data has {X.shape[1]} features")
        
        self.to(device)
        dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                batch_size = X_batch.size(0)
                
                # Update empirical distribution (EMA)
                p_batch = y_batch.float().mean().item()
                new_p = torch.tensor([1-p_batch, p_batch], device=device)
                self.p_emp = self.beta * self.p_emp + (1-self.beta) * new_p
                
                # Update model parameters
                self.optimizer.zero_grad()
                outputs = self(X_batch)
                
                # Calculate weights for DRO loss
                weights = torch.where(
                    y_batch == 1, 
                    self.pi_t[1] / self.p_emp[1],
                    self.pi_t[0] / self.p_emp[0]
                )
                
                # Weighted loss calculation
                loss_per_sample = criterion(outputs, y_batch.float())
                loss = torch.dot(loss_per_sample, weights)
                loss.backward()
                self.optimizer.step()
                
                # Update dual variable pi (using updated model)
                with torch.no_grad():
                    # Get model outputs for the whole batch with updated model
                    f_all = self(X_batch).squeeze()
                    g_pi = torch.zeros(self.n_groups, device=device)
                    
                    # Compute per-sample gradients for pi_t
                    for i in range(batch_size):
                        y_i = y_batch[i]
                        f_i = f_all[i]
                        # Compute z = -pi_t[y_i] * f_i
                        z = -self.pi_t[y_i] * f_i
                        sig = torch.sigmoid(z)
                        # Compute gradient component for this sample
                        grad_pi_i = (sig - y_i.float()) * (-f_i)
                        # Accumulate: 1/batch_size * (1/p_emp[y_i]) * grad_pi_i
                        g_pi[y_i] += (1 / batch_size) * (1 / self.p_emp[y_i]) * grad_pi_i
                    
                    g_pi = torch.clamp(g_pi, max=self.clip_max)
                    
                    # Compute KL divergence between pi_t and p_emp
                    log_pi = torch.log(self.pi_t + 1e-10)
                    kl = (self.pi_t * (log_pi - torch.log(self.p_emp + 1e-10))).sum()
                    
                    # Determine alpha for projection
                    alpha = 0.0 if self.r > kl else 1.0
                    
                    # Update pi_t with normalization
                    numerator = (self.pi_t * (self.p_emp ** alpha)) ** (1/(1+alpha)) * torch.exp(self.eta_pi * g_pi)
                    C = numerator.sum()
                    pi_t_new = numerator / C + self.eps
                    
                # Update dual parameter
                self.pi_t = nn.Parameter(pi_t_new, requires_grad=True)
    
    def predict_proba(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features (n_samples, n_features)
            device: Device for computation
            
        Returns:
            Probability array of shape (n_samples, 2)
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X).float().to(device)
            logits = self(X_tensor)
            proba_1 = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1-proba_1, proba_1]).T
    
    def predict(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Predict binary labels.
        
        Args:
            X: Input features (n_samples, n_features)
            device: Device for computation
            
        Returns:
            Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X, device)[:, 1]
        return (proba > 0.5).astype(int)