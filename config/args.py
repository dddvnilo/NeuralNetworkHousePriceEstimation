import argparse
from typing import List

def parse_args():
    """
    Parse command line arguments for the regression neural network.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train a neural network for regression on Ames Housing dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_path", 
        type=str, 
        default="dataset/AmesHousing.csv",
        help="Path to Ames Housing CSV file"
    )

    parser.add_argument(
        "--target", 
        type=str, 
        default="SalePrice",
        help="Target column name for regression"
    )
    
    parser.add_argument(
        "--drop_missing_thresh", 
        type=float, 
        default=0.30,
        help="Drop columns with more than this fraction of missing values"
    )

    parser.add_argument(
        "--correlation_thresh", 
        type=float, 
        default=0.05,
        help="Remove features with absolute correlation below this threshold"
    )

    parser.add_argument(
        "--pca_components", 
        type=int, 
        default=0,
        help="If >0, apply PCA to reduce features to this many components"
    )
    
    parser.add_argument(
        "--hidden_layers", 
        type=int, 
        nargs="+", 
        default=[128, 64],
        help="Hidden layer sizes (e.g., --hidden_layers 128 64 32)"
    )

    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.0,
        help="Dropout rate for regularization"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Maximum number of training epochs"
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64,
        help="Batch size for training"
    )

    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3,
        help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--patience", 
        type=int, 
        default=12,
        help="Early stopping patience (epochs without improvement)"
    )
    
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.15,
        help="Fraction of data to use for testing"
    )

    parser.add_argument(
        "--val_size", 
        type=float, 
        default=0.15,
        help="Fraction of data to use for validation"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
        help="Directory to save model and results"
    )

    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()