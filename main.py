import os
import random

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.args import parse_args
from data.dataset import TabularDataset
from data.preprocessing import (
    drop_high_missing,
    remove_low_correlation_features,
    split_features_target,
    build_preprocessors,
    transform_dataframe,
    plot_pca_feature_contributions,
    plot_component_importance
)
from models.neural_net import RegressionNet
from training.training import (
    train_model,
    evaluate_model,
    plot_error_distribution
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader

# ---------------------------
# SEED
# ---------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Main pipeline
# ---------------------------

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

    # 3) Drop columns with correlation less then given
    if args.correlation_thresh > 0:
        df = remove_low_correlation_features(df, target_col=args.target, 
                                           threshold=args.correlation_thresh,
                                           output_dir=args.output_dir)
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

    all_feature_names = preprocessors["numeric_cols"] + list(preprocessors["onehot_encoder"].get_feature_names_out(preprocessors["categorical_cols"]))
    plot_pca_feature_contributions(pca, all_feature_names, args.output_dir, top_n=10)

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

    basic_metrics = {
        'mse': float(test_metrics['mse']),
        'rmse': float(test_metrics['rmse']), 
        'mae': float(test_metrics['mae']),
        'r2': float(test_metrics['r2'])
    }
    
    metrics_file = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(basic_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # 13) Save predictions CSV for test set
    preds = test_metrics["preds"]
    trues = test_metrics["trues"]
    out_df = pd.DataFrame({"true": trues, "pred": preds})
    out_csv = os.path.join(args.output_dir, "test_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print("Saved test predictions to", out_csv)

    # 14) scatter plot (true vs pred)
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

    # 15) component importance plot
    plot_component_importance(model, args.output_dir)

    # 16) error distribution + residuals
    plot_error_distribution(out_df, args.output_dir)
    print("Saved error analysis plot to", os.path.join(args.output_dir, "error_analysis.png"))

    print("All done. Outputs in:", args.output_dir)

if __name__ == "__main__":
    main()