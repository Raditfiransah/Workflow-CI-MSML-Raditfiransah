"""
modelling.py — Basic MLflow Tracking (autolog, no tuning)
Jalankan: python modelling.py
MLflow UI: mlflow ui  →  http://localhost:5000
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

# ── Konstanta ──────────────────────────────────────────────────────────────────
TRAIN_PATH  = "Buy_Now_Pay_Later_BNPL_CreditRisk_Dataset_Preprocessing/train_preprocessed.csv"
TEST_PATH   = "Buy_Now_Pay_Later_BNPL_CreditRisk_Dataset_Preprocessing/test_preprocessed.csv"
TARGET      = "default_flag"
EXPERIMENT  = "CreditRisk_Basic"

# ── Load data dengan error handling ────────────────────────────────────────────
try:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Data loaded successfully. Train shape: {train.shape}, Test shape: {test.shape}")
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# ── Split features and target ──────────────────────────────────────────────────
X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

# ── MLflow setup dengan error handling ─────────────────────────────────────────
try:
    # Jika dijalankan via 'mlflow run', experiment sudah ditentukan oleh CLI.
    # set_experiment hanya dipanggil jika dijalankan sebagai script python biasa.
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(EXPERIMENT)
    
    mlflow.sklearn.autolog()
    print(f"MLflow setup complete. Experiment: {EXPERIMENT}")
except Exception as e:
    print(f"Warning: MLflow setup failed - {e}")
    print("Continuing without MLflow tracking...")

# ── Training ───────────────────────────────────────────────────────────────────
# Definisikan model dengan parameter yang jelas
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
)

# Train model & log ke MLflow
with mlflow.start_run() as run:
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    # Save model untuk keperluan artifact & docker
    import shutil
    if os.path.exists("saved_model"):
        shutil.rmtree("saved_model")
    mlflow.sklearn.save_model(model, "saved_model")


# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_proba)

# Print results
print("\n" + "="*50)
print("Run selesai.")
print(f"   Accuracy : {test_accuracy:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall   : {test_recall:.4f}")
print(f"   F1 Score : {test_f1:.4f}")
print(f"   ROC AUC  : {test_roc_auc:.4f}")
print("="*50)

print("\nScript completed successfully.")