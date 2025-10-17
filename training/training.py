import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 10,
    print_every: int = 25
):
    """
    Train the neural network model with early stopping.
    
    Args:
        model: Neural network model to train
        device: Device to train on (CPU/GPU)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data  
        epochs: Maximum number of training epochs
        lr: Learning rate for optimizer
        weight_decay: L2 regularization strength
        patience: Early stopping patience (epochs without improvement)
        print_every: Print metrics every N epochs
        
    Returns:
        model: Trained model with best weights restored
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
            n += Xb.size(0)
        train_epoch_loss = running_loss / n
        train_losses.append(train_epoch_loss)

        model.eval()
        running_val = 0.0
        nval = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                yv = yv.to(device)
                preds_v = model(Xv)
                loss_v = criterion(preds_v, yv)
                running_val += loss_v.item() * Xv.size(0)
                nval += Xv.size(0)
        val_epoch_loss = running_val / nval
        val_losses.append(val_epoch_loss)

        if epoch % print_every == 0:
            print(f"Epoch {epoch:03d}: train_loss={train_epoch_loss:.4f}, val_loss={val_epoch_loss:.4f}")

        if val_epoch_loss < best_val_loss - 1e-6:
            best_val_loss = val_epoch_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping on epoch {epoch}. Best val loss: {best_val_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses

def evaluate_model(model: torch.nn.Module, device: torch.device, loader: torch.utils.data.DataLoader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            out = model(Xb).cpu().numpy().reshape(-1)
            preds.append(out)
            trues.append(yb.numpy().reshape(-1))
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    return {
        "mse": mse, 
        "rmse": rmse, 
        "mae": mae, 
        "r2": r2,
        "preds": preds, 
        "trues": trues
    }

def plot_error_distribution(predictions_df, output_dir: str):
    errors = predictions_df['pred'] - predictions_df['true']
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, alpha=0.7, color='red')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error (Predicted - True)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.scatter(predictions_df['true'], errors, alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('True Price')
    plt.ylabel('Error')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=300)
    plt.close()