import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

class GroupDROClassifier:
    def __init__(
        self,
        n_features,
        n_classes,
        hidden_size=32,
        eta_group=0.1,
        lr=0.01,
        weight_decay=0.01,
        device="cuda",
        random_state=None
    ):
        """
        Group DRO Classifier for class-robust optimization.
        
        Args:
            n_features: Number of input features
            n_classes: Number of classes/groups
            hidden_size: Size of hidden layer (default=32)
            eta_group: Learning rate for group weight updates (default=0.1)
            lr: Learning rate for model optimizer (default=0.01)
            weight_decay: L2 regularization strength (default=0.01)
            device: "cuda" or "cpu" (default="cuda")
            random_state: Random seed for reproducibility
        """
        self.device = device
        self.n_classes = n_classes
        self.eta_group = eta_group
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            
        # Initialize model with dynamic architecture
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        ).to(device)
        
        # Initialize group weights (uniform)
        self.q = torch.ones(n_classes).to(device) / n_classes
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
    def fit(self, X, y, epochs=10, batch_size=32, verbose=True):
        """
        Train model with Group DRO.
        
        Args:
            X: Input features (numpy array or torch.Tensor)
            y: Class labels (numpy array or torch.Tensor)
            epochs: Number of training epochs (default=10)
            batch_size: Batch size (default=32)
            verbose: Print progress (default=True)
        """
        # Convert data to tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
            
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            all_preds = []
            all_targets = []
            
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(X_batch)
                losses = self.criterion(logits, y_batch)
                
                # Compute per-group losses
                group_losses = {}
                for g in range(self.n_classes):
                    group_mask = (y_batch == g)
                    if group_mask.any():
                        group_losses[g] = losses[group_mask].mean()
                
                # Skip batch if no groups present
                if not group_losses:
                    continue
                
                # Update group weights (detached from computation graph)
                with torch.no_grad():
                    q_updated = self.q.clone()
                    for g, loss_val in group_losses.items():
                        # FIX: Use tensor operations instead of float
                        q_updated[g] = self.q[g] * torch.exp(self.eta_group * loss_val)
                    q_updated /= q_updated.sum()
                    self.q = q_updated
                
                # Compute robust loss
                robust_loss = 0
                for g, loss_val in group_losses.items():
                    robust_loss += self.q[g].detach() * loss_val
                
                # Backpropagation
                robust_loss.backward()
                self.optimizer.step()
                
                epoch_loss += robust_loss.item()
                
                # Track predictions for accuracy
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
            
            if verbose:
                acc = accuracy_score(all_targets, all_preds) if all_preds else 0
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | Acc: {acc:.4f}")
                print(f"Group weights: {self.q.cpu().detach().numpy().round(3)}")
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Input features (numpy array or torch.Tensor)
        Returns:
            Predicted class labels (numpy array)
        """
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input features (numpy array or torch.Tensor)
        Returns:
            Class probabilities (numpy array)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            logits = self.model(X)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()