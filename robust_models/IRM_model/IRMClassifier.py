import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from typing import List, Union
import numpy as np

class IRMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        irm_lambda: float = 1.0,
        irm_penalty_anneal_iters: int = 500
    ):
        super().__init__()
        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes))
        
        # Hyperparameters
        self.irm_lambda = irm_lambda
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
        self.learning_rate = learning_rate
        
        # Training state
        self.register_buffer('update_count', torch.tensor(0))
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def _irm_penalty(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(1., device=logits.device).requires_grad_()
        loss1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad1 = autograd.grad(loss1, [scale], create_graph=True)[0]
        grad2 = autograd.grad(loss2, [scale], create_graph=True)[0]
        return torch.sum(grad1 * grad2)

    def _update(self, minibatches: List[torch.Tensor]):
        # Calculate penalty weight
        penalty_weight = self.irm_lambda if self.update_count >= self.irm_penalty_anneal_iters else 1.0
        
        # Process all minibatches
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self(all_x)
        all_logits_idx = 0
        
        nll, penalty = 0., 0.
        for x, y in minibatches:
            logits = all_logits[all_logits_idx:all_logits_idx + len(x)]
            all_logits_idx += len(x)
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        
        # Average losses
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        
        # Reset optimizer when reaching anneal iterations
        if self.update_count == self.irm_penalty_anneal_iters:
            self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        
        return loss.item()

    def fit(
        self,
        domain_loaders: List[DataLoader],
        n_iterations: int = 1000
    ):
        self.train()
        iterators = [iter(loader) for loader in domain_loaders]
        
        for _ in range(n_iterations):
            minibatches = []
            for i in range(len(iterators)):
                try:
                    x, y = next(iterators[i])
                except StopIteration:
                    # Reset exhausted iterator
                    iterators[i] = iter(domain_loaders[i])
                    x, y = next(iterators[i])
                
                minibatches.append((
                    x.to(self.device).float(),
                    y.to(self.device).long()
                ))
            
            self._update(minibatches)

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        self.eval()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        
        with torch.no_grad():
            logits = self(X)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions.cpu().numpy()