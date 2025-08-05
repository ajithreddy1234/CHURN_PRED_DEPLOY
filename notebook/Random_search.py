import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from sklearn.metrics import f1_score

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_search(
        model_class,
        model_init_fn,
        param_dist,
        X_train, y_train,
        X_val, y_val,
        device=device,
        n_iter=10,
        train_epochs=10,
        patience=3,
        criterion_fn=nn.BCELoss,
        metric_fn=f1_score,
        threshold=0.5,
        verbose=True
):
    """
    Random search for hyperparameters for any ANN model.

    Args:
        model_class: The PyTorch nn.Module class to instantiate.
        model_init_fn: Function that takes sampled hyperparams and returns model init kwargs.
        param_dist: Dict of hyperparameter lists to sample from.
        X_train, y_train: Training tensors.
        X_val, y_val: Validation tensors.
        device: 'cpu' or 'cuda'.
        n_iter: Number of random search iterations.
        train_epochs: Max epochs per search iteration.
        patience: Early stopping patience.
        criterion_fn: Loss function class.
        metric_fn: Scoring function (e.g., f1_score).
        threshold: Threshold for binary predictions.
        verbose: If True, print progress.
    Returns:
        best_params: Dict of best hyperparameters.
    """
    best_score = 0
    best_params = None

    for i in range(n_iter):
        # Sample hyperparameters
        sampled_params = {k: random.choice(v) for k, v in param_dist.items()}
        model_kwargs = model_init_fn(sampled_params)
        model = model_class(**model_kwargs).to(device)
        criterion = criterion_fn()
        optimizer = optim.Adam(model.parameters(), lr=sampled_params.get('lr', 1e-3),
                               weight_decay=sampled_params.get('weight_decay', 0))

        # Data loaders
        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_data, batch_size=sampled_params.get('batch_size', 32), shuffle=True)
        val_loader = DataLoader(val_data, batch_size=sampled_params.get('batch_size', 32))

        # Early stopping
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(train_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch.squeeze())
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch.squeeze())
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        # Evaluate metric
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                preds = (outputs >= threshold).float().cpu().numpy()
                labels = y_batch.squeeze().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        score = metric_fn(all_labels, all_preds)
        if verbose:
            print(f"Iteration {i + 1}: Score={score:.4f} | Params: {sampled_params}")
        if score > best_score:
            best_score = score
            best_params = sampled_params.copy()
    if verbose:
        print("\nBest Params:", best_params)
        print("Best Score:", best_score)
    return best_params
