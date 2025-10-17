from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict

def drop_high_missing(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Drop columns with more than `threshold` fraction missing values.
    """
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()
    print(f"Dropping {len(to_drop)} columns with >{threshold*100:.0f}% missing values.")
    return df.drop(columns=to_drop)

def remove_low_correlation_features(df: pd.DataFrame, target_col: str, threshold: float = 0.05, output_dir: str = "output") -> pd.DataFrame:
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

    # Display correlations before and after
    plt.figure(figsize=(12, 6))
    
    # All correlations
    all_correlations = correlations[correlations.index != target_col]
    
    # Split into high and low correlations
    high_corr = all_correlations[all_correlations >= threshold]
    low_corr = all_correlations[all_correlations < threshold]
    
    # Graphical display
    plt.subplot(1, 2, 1)
    plt.barh(range(len(high_corr)), high_corr.sort_values().values)
    plt.yticks(range(len(high_corr)), high_corr.sort_values().index)
    plt.title(f'Retained Features (correlation ≥ {threshold})')
    plt.xlabel('Absolute Correlation')
    
    plt.subplot(1, 2, 2)
    plt.barh(range(len(low_corr)), low_corr.sort_values().values)
    plt.yticks(range(len(low_corr)), low_corr.sort_values().index)
    plt.title(f'Removed Features (correlation < {threshold})')
    plt.xlabel('Absolute Correlation')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
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