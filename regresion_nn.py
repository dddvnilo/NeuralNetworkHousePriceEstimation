import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------
# Utilities & Dataset class
# ---------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Preprocessing pipeline
# ---------------------------

def drop_high_missing(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Drop columns with more than `threshold` fraction missing values.
    """
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()
    print(f"Dropping {len(to_drop)} columns with >{threshold*100:.0f}% missing values.")
    return df.drop(columns=to_drop)

def remove_low_correlation_features(df: pd.DataFrame, target_col: str, threshold: float = 0.05) -> pd.DataFrame:
    """
    Remove features with absolute correlation with target below threshold.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        return df
    
    correlations = numeric_df.corr()[target_col].abs()
    low_corr_cols = correlations[correlations < threshold].index.tolist()
    low_corr_cols = [col for col in low_corr_cols if col != target_col]
    
    print(f"Removing {len(low_corr_cols)} features with correlation < {threshold}")
    return df.drop(columns=low_corr_cols)

def split_features_target(df: pd.DataFrame, target_col: str = "SalePrice") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split target feature from other features.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    return X, y

def build_preprocessors(
    X_train: pd.DataFrame,
    numeric_strategy: str = "median"
):
    """
    Fit imputers, encoders and scalers on training data only. Return fitted transformers.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Numeric cols: {len(numeric_cols)}, Categorical cols: {len(categorical_cols)}")

    num_imputer = SimpleImputer(strategy=numeric_strategy)
    num_imputer.fit(X_train[numeric_cols])

    # For categoricals: fillna with special token then ordinal encode.
    cat_fill_value = "Missing"
    X_train_cat = X_train[categorical_cols].fillna(cat_fill_value)

    # OrdinalEncoder with handle_unknown via setting unknown_value
    onehot_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    onehot_enc.fit(X_train_cat)

    scaler = StandardScaler()
    # We'll fit scaler on numeric features after imputation
    X_train_num_imputed = num_imputer.transform(X_train[numeric_cols])
    scaler.fit(X_train_num_imputed)

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "num_imputer": num_imputer,
        "onehot_encoder": onehot_enc,
        "cat_fill_value": cat_fill_value,
        "scaler": scaler
    }

def transform_dataframe(df: pd.DataFrame, preprocessors: dict) -> np.ndarray:
    """
    Apply fitted preprocessors to a dataframe and return numpy array (features).
    """
    ncols = preprocessors["numeric_cols"]
    ccols = preprocessors["categorical_cols"]
    num_imputer = preprocessors["num_imputer"]
    onehot_enc = preprocessors["onehot_encoder"]
    cat_fill_value = preprocessors["cat_fill_value"]
    scaler = preprocessors["scaler"]

    # Numeric
    X_num = df[ncols]
    X_num_imputed = num_imputer.transform(X_num)

    # Scale
    X_num_scaled = scaler.transform(X_num_imputed)

    # Categorical
    if len(ccols) > 0:
        X_cat = df[ccols].fillna(cat_fill_value)
        X_cat_encoded = onehot_enc.transform(X_cat)
        X_total = np.hstack([X_num_scaled, X_cat_encoded])
    else:
        X_total = X_num_scaled

    return X_total

# ---------------------------
# Model
# ---------------------------

class RegressionNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        last_dim = input_dim
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))  # output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Training & Evaluation
# ---------------------------

def train_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 10,
    print_every: int = 1
):
    criterion = nn.MSELoss()
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

        # Validation
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

        # early stopping
        if val_epoch_loss < best_val_loss - 1e-6:
            best_val_loss = val_epoch_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping on epoch {epoch}. Best val loss: {best_val_loss:.6f}")
                break

    # load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses

def evaluate_model(model: nn.Module, device: torch.device, loader: DataLoader):
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

# ---------------------------
# Main pipeline
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="dataset/AmesHousing.csv", help="Path to Ames CSV file")
    p.add_argument("--target", type=str, default="SalePrice")
    p.add_argument("--drop_missing_thresh", type=float, default=0.30,
                   help="Drop columns with more than this fraction missing")
    p.add_argument("--correlation_thresh", type=float, default=0.05,
                   help="Remove features with absolute correlation below this threshold")
    p.add_argument("--pca_components", type=int, default=0,
                   help="If >0, apply PCA to reduce features to this many components")
    p.add_argument("--hidden_layers", type=int, nargs="+", default=[128, 64],
                   help="Hidden layer sizes")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--output_dir", type=str, default="output")
    p.add_argument("--random_state", type=int, default=SEED)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = set_device()
    print(f"Device: {device}")

    # 1) Load data
    df = pd.read_csv(args.data_path)
    print("Loaded dataset shape:", df.shape)

    # 2) Drop columns with too many missing values
    df = drop_high_missing(df, threshold=args.drop_missing_thresh)
    print("After dropping columns:", df.shape)

    if args.correlation_thresh > 0:
        df = remove_low_correlation_features(df, target_col=args.target, 
                                           threshold=args.correlation_thresh)
        print("After removing low-correlation features:", df.shape)

    # 4) split X/y and then train/val/test (we must split BEFORE fitting preprocessors)
    X, y = split_features_target(df, target_col=args.target)

    # First split off test
    test_size = args.test_size
    val_size = args.val_size
    # Combine val+test fraction relative splitting
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=args.random_state
    )
    # Now split X_temp into train and val
    # val fraction of original = val_size -> fraction of X_temp = val_size / (1 - test_size)
    val_fraction_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=args.random_state
    )

    print(f"Sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # 5) Fit preprocessors on train only
    preprocessors = build_preprocessors(X_train)

    # 6) Transform datasets
    X_train_np = transform_dataframe(X_train, preprocessors)
    X_val_np = transform_dataframe(X_val, preprocessors)
    X_test_np = transform_dataframe(X_test, preprocessors)

    # 7) PCA
    if args.pca_components and args.pca_components > 0:
        pca = PCA(n_components=args.pca_components, random_state=args.random_state)
        pca.fit(X_train_np)
        X_train_np = pca.transform(X_train_np)
        X_val_np = pca.transform(X_val_np)
        X_test_np = pca.transform(X_test_np)
        print(f"PCA applied: new feature dim {X_train_np.shape[1]}")
    else:
        print(f"No PCA. Feature dim: {X_train_np.shape[1]}")

    # 8) Build dataloaders
    train_ds = TabularDataset(X_train_np, y_train)
    val_ds = TabularDataset(X_val_np, y_val)
    test_ds = TabularDataset(X_test_np, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 9) Create model
    input_dim = X_train_np.shape[1]
    model = RegressionNet(input_dim=input_dim, hidden_layers=args.hidden_layers, dropout=args.dropout)
    print("Model:", model)

    # 10) Train
    model, train_losses, val_losses = train_model(
        model, device, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience
    )

    # 11) Save training plot
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(args.output_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print("Saved loss curve to", loss_plot_path)
    plt.close()

    # 12) Evaluate on test
    test_metrics = evaluate_model(model, device, test_loader)
    print("Test metrics:")
    print(f"  MSE:  {test_metrics['mse']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}") 

    # 13) Save model
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_layers": args.hidden_layers,
        "preprocessors": preprocessors
    }, model_path)
    print("Saved model to", model_path)

    # 14) Save predictions CSV for test set
    preds = test_metrics["preds"]
    trues = test_metrics["trues"]
    out_df = pd.DataFrame({"true": trues, "pred": preds})
    out_csv = os.path.join(args.output_dir, "test_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print("Saved test predictions to", out_csv)

    # 15) scatter plot (true vs pred)
    plt.figure(figsize=(6,6))
    plt.scatter(trues, preds, alpha=0.6)
    minv = min(trues.min(), preds.min())
    maxv = max(trues.max(), preds.max())
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("True SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("True vs Predicted on test set")
    plt.tight_layout()
    scatter_path = os.path.join(args.output_dir, "true_vs_pred.png")
    plt.savefig(scatter_path)
    print("Saved scatter to", scatter_path)
    plt.close()

    print("All done. Outputs in:", args.output_dir)

if __name__ == "__main__":
    main()