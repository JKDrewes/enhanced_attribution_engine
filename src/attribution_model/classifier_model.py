"""
Trains a logistic regression model to predict conversion using merged structured
and LLM-derived features.
"""

import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logger import logger
import bootstrap

# --- Paths ---
FEATURES_FILE = bootstrap.PROCESSED_DIR / "model_input_features.csv"
OUTPUT_DIR = bootstrap.OUTPUTS_DIR
MODEL_FILE = OUTPUT_DIR / "conversion_model.pkl"
PREDICTIONS_FILE = OUTPUT_DIR / "conversion_predictions.csv"

# --- Load merged features ---
logger.info(f"Loading merged features from {FEATURES_FILE}...")
df_features = pd.read_csv(FEATURES_FILE)

# --- Define target and predictors ---
TARGET = "conversion"
feature_cols = [
    col for col in df_features.columns
    if col not in ["user_id", "conversion", "revenue"]
]

X = df_features[feature_cols]
y = df_features[TARGET]

# --- Handle datetime-like columns ---
for col in X.columns:
    if X[col].dtype == "object":
        try:
            X.loc[:, col] = pd.to_datetime(X[col], errors="raise")
            logger.info(f"Converted {col} to datetime")
        except Exception:
            # Not a datetime column, leave as-is
            pass

# --- Extract numeric features from datetime columns ---
datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns
for col in datetime_cols:
    X.loc[:, f"{col}_hour"] = X[col].dt.hour
    X.loc[:, f"{col}_dayofweek"] = X[col].dt.dayofweek
    X.loc[:, f"{col}_month"] = X[col].dt.month
    X = X.drop(columns=[col])

# --- Drop rows with any missing values ---
before_drop = len(X)
X = X.dropna()
y = y.loc[X.index]  # keep y aligned with X
logger.info(f"Dropped {before_drop - len(X)} rows with missing values.")

logger.info(f"Target distribution before split:\n{y.value_counts()}")

# --- Train/test split (stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
logger.info(f"Train target distribution:\n{y_train.value_counts()}")
logger.info(f"Test target distribution:\n{y_test.value_counts()}")

# --- Identify column types ---
num_cols = X_train.select_dtypes(include=["number"]).columns
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

logger.info(f"Numeric columns: {list(num_cols)}")
logger.info(f"Categorical columns: {list(cat_cols)}")

# --- Preprocessing ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# --- Build pipeline ---
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# --- Train model ---
logger.info("Training LogisticRegression model with preprocessing pipeline...")
model.fit(X_train, y_train)

# --- Evaluate model ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

logger.info(f"Accuracy: {acc:.4f}")
logger.info(f"ROC AUC: {auc:.4f}")

# --- Save model ---
joblib.dump(model, MODEL_FILE)
logger.info(f"Model saved to {MODEL_FILE}")

# --- Save predictions ---
df_preds = X_test.copy()
df_preds["user_id"] = df_features.loc[X_test.index, "user_id"]
df_preds["y_true"] = y_test
df_preds["y_pred"] = y_prob
df_preds.to_csv(PREDICTIONS_FILE, index=False)
logger.info(f"Predictions saved to {PREDICTIONS_FILE}")

if __name__ == "__main__":
    logger.info("Running classifier model training directly")
