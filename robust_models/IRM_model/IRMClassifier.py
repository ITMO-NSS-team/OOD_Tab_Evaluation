import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings

class UniversalIRM:
    def __init__(self, task='classification', n_environments=2, lambda_irm=0.5,
                 hidden_dim=64, lr=0.001, epochs=100, batch_size=32, 
                 patience=5, device=None, verbose=True):
        """
        Universal Invariant Risk Minimization (IRM) for tabular data
        
        Parameters:
        task (str): 'classification' or 'regression'
        n_environments (int): Number of environments to create
        lambda_irm (float): IRM regularization strength
        hidden_dim (int): Number of units in hidden layers
        lr (float): Learning rate
        epochs (int): Maximum training epochs
        batch_size (int): Batch size
        patience (int): Early stopping patience
        device (str): 'cuda' or 'cpu'
        verbose (bool): Print training progress
        """
        self.task = task
        self.n_environments = n_environments
        self.lambda_irm = lambda_irm
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.optimizer = None
        self.best_loss = float('inf')
        self.counter = 0
        self.is_fitted = False
        
    def _create_environments(self, X, y):
        """
        Automatically create environments using feature clustering
        """
        if self.n_environments == 1:
            return [(X, y)]
        
        # Use k-means clustering on features to create environments
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_environments, random_state=42, n_init=10)
        env_labels = kmeans.fit_predict(X)
        
        environments = []
        for i in range(self.n_environments):
            env_mask = (env_labels == i)
            env_X = X[env_mask]
            env_y = y[env_mask]
            
            # Skip empty environments
            if len(env_X) > 0:
                environments.append((env_X, env_y))
        
        # Handle case where we got fewer environments than requested
        if len(environments) < self.n_environments:
            warnings.warn(f"Created only {len(environments)} environments instead of {self.n_environments}")
            self.n_environments = len(environments)
            
        return environments
    
    def _build_model(self, input_dim):
        """Create neural network architecture"""
        if self.task == 'classification':
            output_dim = len(np.unique(self.y_train)) if len(np.unique(self.y_train)) > 2 else 1
        else:
            output_dim = 1
        
        model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim))
        
        # For binary classification
        if self.task == 'classification' and output_dim == 1:
            model.add_module('sigmoid', nn.Sigmoid())
        
        return model.to(self.device)
    
    def _irm_loss(self, model, x, y):
        """Calculate IRM loss with gradient penalty"""
        # Standard task loss
        outputs = model(x)
        
        if self.task == 'classification':
            if outputs.shape[1] == 1:  # Binary classification
                loss = nn.BCELoss()(outputs.squeeze(), y.float())
            else:  # Multiclass classification
                loss = nn.CrossEntropyLoss()(outputs, y)
        else:  # Regression
            loss = nn.MSELoss()(outputs.squeeze(), y)
        
        # IRM penalty: Gradient penalty w.r.t. dummy scale parameter
        w = torch.tensor(1.0, requires_grad=True, device=self.device)
        
        # Create a feature extractor (first part of the network)
        features = x
        for i, layer in enumerate(model):
            if i == len(model) - 1:  # Stop before last layer
                break
            features = layer(features)
        
        # Apply scaling to features
        scaled_features = w * features
        
        # Pass through classifier (last layer)
        if self.task == 'classification' and model[-1].__class__.__name__ == 'Sigmoid':
            # For binary classification with sigmoid
            classifier = model[-2]
            logits_scale = classifier(scaled_features)
            logits_scale = model[-1](logits_scale)  # Apply sigmoid
        elif self.task == 'classification':
            # Multiclass classification
            classifier = model[-1]
            logits_scale = classifier(scaled_features)
        else:
            # Regression
            classifier = model[-1]
            logits_scale = classifier(scaled_features)
        
        # Compute loss on scaled features
        if self.task == 'classification':
            if logits_scale.shape[1] == 1:  # Binary
                loss_scale = nn.BCELoss()(logits_scale.squeeze(), y.float())
            else:  # Multiclass
                loss_scale = nn.CrossEntropyLoss()(logits_scale, y)
        else:  # Regression
            loss_scale = nn.MSELoss()(logits_scale.squeeze(), y)
        
        # Compute gradient of loss w.r.t. w at w=1.0
        grad_w = torch.autograd.grad(loss_scale, w, create_graph=True)[0]
        penalty = torch.sum(grad_w ** 2)
        
        return loss + self.lambda_irm * penalty
    
    def fit(self, X, y, envs=None):
        """
        Train IRM model
        
        Parameters:
        X (array-like): Input features
        y (array-like): Target values
        envs (array-like): Predefined environment labels (optional)
        """
        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Store for model building
        self.X_train = X
        self.y_train = y
        
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create environments
        if envs is None:
            environments = self._create_environments(X_scaled, y)
        else:
            envs = np.array(envs)
            environments = []
            for env_id in np.unique(envs):
                env_mask = (envs == env_id)
                environments.append((X_scaled[env_mask], y[env_mask]))
        
        if len(environments) < 2:
            warnings.warn("IRM requires at least 2 environments. Using single environment without IRM penalty.")
            self.lambda_irm = 0.0
        
        # Initialize model
        input_dim = X_scaled.shape[1]
        self.model = self._build_model(input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Create data loaders
        loaders = []
        for env_X, env_y in environments:
            # Convert to tensors
            X_tensor = torch.tensor(env_X, dtype=torch.float32)
            y_tensor = torch.tensor(env_y, dtype=torch.long if self.task == 'classification' else torch.float32)
            
            # Create dataset and loader
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            loaders.append(loader)
        
        # Training loop
        self.best_loss = float('inf')
        self.counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Create iterators for each environment
            iter_loaders = [iter(loader) for loader in loaders]
            min_batches = min(len(loader) for loader in loaders)
            
            for _ in range(min_batches):
                losses = []
                for loader in iter_loaders:
                    try:
                        x_batch, y_batch = next(loader)
                    except StopIteration:
                        continue
                        
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    self.optimizer.zero_grad()
                    loss = self._irm_loss(self.model, x_batch, y_batch)
                    losses.append(loss)
                
                # Average loss across environments
                avg_loss = sum(losses) / len(losses)
                avg_loss.backward()
                self.optimizer.step()
                total_loss += avg_loss.item()
                num_batches += 1
            
            # Calculate average epoch loss
            if num_batches > 0:
                epoch_loss = total_loss / num_batches
            else:
                epoch_loss = float('inf')
            
            # Early stopping check
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict()
            else:
                self.counter += 1
            
            # Print progress
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch+1}/{self.epochs}: Loss = {epoch_loss:.4f}")
            
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict outputs for input samples"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        X = np.array(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            
            if self.task == 'classification':
                if outputs.shape[1] == 1:  # Binary classification
                    preds = (outputs.squeeze() > 0.5).long()
                else:  # Multiclass classification
                    _, preds = torch.max(outputs, 1)
            else:  # Regression
                preds = outputs.squeeze()
                
        return preds.cpu().numpy()
    
    def predict_proba(self, X):
        """Predict class probabilities (classification only)"""
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
            
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        X = np.array(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            
            if outputs.shape[1] == 1:  # Binary classification
                probs = torch.sigmoid(outputs).squeeze()
                return np.vstack([1 - probs.cpu().numpy(), probs.cpu().numpy()]).T
            else:  # Multiclass classification
                return torch.softmax(outputs, dim=1).cpu().numpy()
    
    def score(self, X, y):
        """Calculate model score (accuracy for classification, R2 for regression)"""
        preds = self.predict(X)
        y = np.array(y)
        
        if self.task == 'classification':
            return accuracy_score(y, preds)
        else:
            return r2_score(y, preds)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_diabetes
    
    # ====================
    # Classification Example
    # ====================
    print("\n" + "="*50)
    print("Classification Example: Breast Cancer Dataset")
    print("="*50)
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train IRM classifier
    irm_clf = UniversalIRM(task='classification', n_environments=3, 
                          lambda_irm=0.8, epochs=100, verbose=True)
    irm_clf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = irm_clf.score(X_train, y_train)
    test_acc = irm_clf.score(X_test, y_test)
    print(f"\nIRM Classifier Accuracy:")
    print(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    # ====================
    # Regression Example
    # ====================
    print("\n" + "="*50)
    print("Regression Example: Diabetes Dataset")
    print("="*50)
    
    # Load data
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train IRM regressor
    irm_reg = UniversalIRM(task='regression', n_environments=3, 
                          lambda_irm=0.5, epochs=200, verbose=True)
    irm_reg.fit(X_train, y_train)
    
    # Evaluate
    train_r2 = irm_reg.score(X_train, y_train)
    test_r2 = irm_reg.score(X_test, y_test)
    print(f"\nIRM Regressor R2 Score:")
    print(f"Train: {train_r2:.4f}, Test: {test_r2:.4f}")