# TEAM 34 - COMP-432 Kaggle Competition Project (PyTorch Version)
# Seungjoon Lee         (40106614)
# Maxwell   Li          (40210708)
# Jeremy    Anderson    (40094989)

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 200)

# -------- CONFIG ----------
COMPETITION     = "machine-learning-comp-432-project"
LOCAL_DATA_DIR  = Path("./data")
RANDOM_STATE    = 42
N_FOLDS         = 5
TARGET_COL      = None  # will be inferred as 'label'
ID_COL          = None  # will be inferred as 'id'
# --------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# --------------------- Data Loading ---------------------
def load_data():
    # attempt to find train/test files
    possible_train  = list(LOCAL_DATA_DIR.glob("*train*.csv"))
    possible_test   = list(LOCAL_DATA_DIR.glob("*test*.csv"))
    possible_sub    = list(LOCAL_DATA_DIR.glob("*sample*.csv"))
    if len(possible_train) == 0 or len(possible_test) == 0:
        # try src/ as fallback
        possible_train  = list(Path("src").glob("*train*.csv"))
        possible_test   = list(Path("src").glob("*test*.csv"))
        possible_sub    = list(Path("src").glob("*sample*.csv"))

    if len(possible_train) == 0:
        raise FileNotFoundError("train file not found. Put train.csv in ./data or ./src.")

    train_path      = possible_train[0]
    test_path       = possible_test[0] if len(possible_test) else None
    sample_sub_path = possible_sub[0] if len(possible_sub) else None

    print("Train path:", train_path)
    print("Test path :", test_path)
    print("Sample sub:", sample_sub_path)

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path) if test_path is not None else None
    sample_sub = pd.read_csv(sample_sub_path) if sample_sub_path is not None else None

    return df_train, df_test, sample_sub

# --------------------- Column Inference ---------------------
def infer_columns(df_train: pd.DataFrame):
    global TARGET_COL, ID_COL

    # Target: try typical names first
    if TARGET_COL is None:
        for cand in ["label", "target", "TARGET", "y", "Y"]:
            if cand in df_train.columns:
                TARGET_COL = cand
                break

    # If still None, fall back to last numeric column
    if TARGET_COL is None:
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found to use as target.")
        TARGET_COL = numeric_cols[-1]

    # ID column: try common patterns
    if ID_COL is None:
        for cand in ["id", "Id", "ID", "index", "row_id"]:
            if cand in df_train.columns:
                ID_COL = cand
                break

    print("Inferred TARGET_COL:", TARGET_COL, "ID_COL:", ID_COL)

# --------------------- Task Detection ---------------------
def detect_task(y: pd.Series):
    """
    For this COMP432 project, the task is always multiclass classification (50 classes).
    We force 'classification' to avoid the previous mis-detection as regression.
    """
    return "classification"

# --------------------- Quick EDA ---------------------
def quick_eda(df: pd.DataFrame):
    print("Shape:", df.shape)
    print("Missing per column (top 10):")
    print(df.isna().mean().sort_values(ascending=False).head(10))
    print("Dtypes:")
    print(df.dtypes.value_counts())
    print(df.describe().T.head(5))

# --------------------- Preprocessing ---------------------
def build_pipelines(df_train: pd.DataFrame):
    """
    Simpler version tailored for this competition:
    - All feature_* columns are numeric.
    - We standard-scale numeric features.
    Returns:
        preprocessor: ColumnTransformer
        numeric_cols: list of numeric feature columns
        small_card: empty list (kept for compatibility)
        large_card: empty list (kept for compatibility)
    """
    numeric_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # ensure we do not include the target column if present
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    cat_cols = []  # no categorical features in this competition

    print("Numeric cols:", len(numeric_cols), "Cat cols:", len(cat_cols))

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
        ],
        remainder="drop"
    )

    small_card = []
    large_card = []

    return preprocessor, numeric_cols, small_card, large_card

# --------------------- PyTorch Model ---------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

# --------------------- Training & Prediction ---------------------
def train_and_predict(df_train: pd.DataFrame, df_test: pd.DataFrame | None = None, sample_sub: pd.DataFrame | None = None):
    infer_columns(df_train)
    y = df_train[TARGET_COL]
    task = detect_task(y)
    print("Detected task:", task)

    # Features only (drop target)
    X = df_train.drop(columns=[TARGET_COL])
    X_test = df_test.copy() if df_test is not None else None

    quick_eda(X)

    preprocessor, numeric_cols, small_card, large_card = build_pipelines(df_train)

    # KFold / StratifiedKFold
    if task == "classification" and y.nunique() > 1:
        folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    else:
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Prepare arrays
    oof_preds = np.zeros(len(df_train), dtype=int)
    feature_importances = pd.DataFrame()  # kept for API compatibility

    num_classes = y.nunique()
    if X_test is not None:
        # We will use majority voting across folds
        test_votes = np.zeros((len(X_test), num_classes), dtype=int)
    else:
        test_votes = None

    for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y if task == "classification" else None)):
        print(f"===== FOLD {fold+1}/{N_FOLDS} =====")
        X_tr, X_val = X.iloc[tr_idx].reset_index(drop=True), X.iloc[val_idx].reset_index(drop=True)
        y_tr, y_val = y.iloc[tr_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)

        # Fit preprocessor on train fold
        preprocessor.fit(X_tr)
        X_tr_proc = preprocessor.transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)
        if X_test is not None:
            X_test_proc = preprocessor.transform(X_test)
        else:
            X_test_proc = None

        input_dim = X_tr_proc.shape[1]
        print(f"Input dim after preprocessing: {input_dim}")

        # Build PyTorch datasets and loaders
        train_dataset = NumpyDataset(X_tr_proc, y_tr.values)
        val_dataset = NumpyDataset(X_val_proc, y_val.values)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        # Model, loss, optimizer
        model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_acc = 0.0
        best_state = None
        n_epochs = 20

        for epoch in range(1, n_epochs + 1):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    logits = model(xb)
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(yb.cpu().numpy())
            val_acc = accuracy_score(val_labels, val_preds)
            avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            print(f"Epoch {epoch:02d}/{n_epochs}  Train Loss: {avg_train_loss:.4f}  Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()

        print(f"Best Val Acc for fold {fold+1}: {best_val_acc:.4f}")

        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final validation predictions (for OOF)
        model.eval()
        val_dataset_final = NumpyDataset(X_val_proc, y_val.values)
        val_loader_final = DataLoader(val_dataset_final, batch_size=512, shuffle=False)

        final_val_preds = []
        with torch.no_grad():
            for xb, yb in val_loader_final:
                xb = xb.to(DEVICE)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                final_val_preds.extend(preds.cpu().numpy())

        final_val_preds = np.array(final_val_preds, dtype=int)
        oof_preds[val_idx] = final_val_preds

        # Test predictions for this fold (majority vote)
        if X_test_proc is not None and test_votes is not None:
            test_dataset = NumpyDataset(X_test_proc, None)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

            fold_test_preds = []
            with torch.no_grad():
                for xb in test_loader:
                    xb = xb.to(DEVICE)
                    logits = model(xb)
                    preds = torch.argmax(logits, dim=1)
                    fold_test_preds.extend(preds.cpu().numpy())
            fold_test_preds = np.array(fold_test_preds, dtype=int)
            rows = np.arange(len(fold_test_preds))
            test_votes[rows, fold_test_preds] += 1

    # OOF accuracy
    overall_acc = accuracy_score(y, oof_preds)
    print("Overall OOF Accuracy:", overall_acc)

    # Derive final test predictions by majority vote
    if test_votes is not None:
        test_preds = np.argmax(test_votes, axis=1)
    else:
        test_preds = None

    # Save a simple PyTorch model (last fold)
    torch.save(model.state_dict(), "baseline_mlp_model.pt")
    print("Saved PyTorch model to baseline_mlp_model.pt")

    # Create submission
    if df_test is not None and sample_sub is not None and test_preds is not None:
        submission = sample_sub.copy()
        if "label" in submission.columns:
            submission["label"] = test_preds
        else:
            # heuristics: find prediction column in sample_sub
            pred_cols = [c for c in submission.columns if c not in (ID_COL,)]
            if len(pred_cols) == 1:
                submission[pred_cols[0]] = test_preds
            elif len(pred_cols) >= 1:
                submission[pred_cols[0]] = test_preds
            else:
                submission["prediction"] = test_preds

        submission.to_csv("submission_baseline.csv", index=False)
        print("Wrote submission_baseline.csv")

    return model, oof_preds, test_preds, feature_importances

# --- Main execution ---
if __name__ == "__main__":
    df_train, df_test, sample_sub = load_data()
    model, oof, test_preds, feature_importances = train_and_predict(df_train, df_test, sample_sub)

