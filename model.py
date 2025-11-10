# TEAM 34 - COMP-432 Kaggle Competition Project
# Seungjoon Lee         (40106614)
# Maxwell   Li          (40210708)
# Jeremy    Anderson    (40094989)
#
# Requirements:
#   pip install -U scikit-learn pandas numpy matplotlib seaborn lightgbm xgboost category_encoders optuna

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 200)

# -------- CONFIG ----------
COMPETITION     = "machine-learning-comp-432-project"  # change if you want to use Kaggle API
LOCAL_DATA_DIR  = Path("./data")  # where train.csv/test.csv live if not using Kaggle API
RANDOM_STATE    = 42
N_FOLDS         = 5
TARGET_COL      = None  # if you know the target column name, set it here; otherwise we'll try to infer
ID_COL          = None      # if you know ID column set here (e.g., 'Id' or 'PassengerId')
# --------------------------

# Load dataset
def load_data():
    # attempt to find train/test files
    possible_train  = list(LOCAL_DATA_DIR.glob("*train*.csv"))
    possible_test   = list(LOCAL_DATA_DIR.glob("*test*.csv"))
    possible_sub    = list(LOCAL_DATA_DIR.glob("*sample*.csv"))
    if len(possible_train) == 0 or len(possible_test) == 0:
        # try root
        possible_train  = list(Path("src").glob("*train*.csv"))
        possible_test   = list(Path("src").glob("*test*.csv"))
        possible_sub    = list(Path("src").glob("*sample*.csv"))

    if len(possible_train) == 0:
        raise FileNotFoundError("train file not found. Put train.csv in ./data or set path.")

    train_path      = possible_train[0]
    test_path       = possible_test[0] if len(possible_test) else None
    sample_sub_path = possible_sub[0] if len(possible_sub) else None

    print("Using train:", train_path, "test:", test_path, "sample_submission:", sample_sub_path)

    df_train    = pd.read_csv(train_path)
    df_test     = pd.read_csv(test_path) if test_path else None
    sample_sub  = pd.read_csv(sample_sub_path) if sample_sub_path else None

    return df_train, df_test, sample_sub

# Try to infer target and id columns
def infer_columns(df_train):
    global TARGET_COL, ID_COL

    if TARGET_COL is None:
        # heuristics: if there's a column named 'target' or 'label' or last numeric column?
        for cand in ['target', 'label', 'TARGET', 'y', 'Y', 'SalePrice', 'Survived']:
            if cand in df_train.columns:
                TARGET_COL = cand
                break

        if TARGET_COL is None:
            # choose non-unique low-cardinality numeric or binary column
            numeric_cols    = df_train.select_dtypes(include=[np.number]).columns.tolist()
            # pick last numeric column excluding id-like columns
            id_like         = [c for c in df_train.columns if c.lower() in ('id','index','row_id','index_id')]
            possible        = [c for c in numeric_cols if c not in id_like]
            if len(possible) >= 1:
                TARGET_COL = possible[-1]

    if ID_COL is None:
        for cand in ['Id','ID','id','index','row_id','PassengerId']:
            if cand in df_train.columns:
                ID_COL = cand
                break

    print("Inferred TARGET_COL:", TARGET_COL, "ID_COL:", ID_COL)

# Detect task: classification or regression
def detect_task(y):
    if y.dtype == 'O':
        return 'classification'

    # if binary or small number of unique ints -> classification
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
        return 'classification'

    # otherwise regression
    if pd.api.types.is_float_dtype(y) or pd.api.types.is_integer_dtype(y):
        if y.nunique() > 20:
            return 'regression'

    return 'regression'

# Basic EDA quick report
def quick_eda(df):
    print("Shape:", df.shape)
    print("Missing per column:\n", df.isna().mean().sort_values(ascending=False).head(20))
    print("Dtypes:\n", df.dtypes.value_counts())
    print(df.describe(include='all').T.head(30))

# Simple preprocessing pipeline builder
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.cols]

def build_pipelines(df_train, categorical_threshold=0.05):
    # decide categorical vs numerical
    n_rows          = df_train.shape[0]
    numeric_cols    = df_train.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols        = df_train.select_dtypes(include=['object','category']).columns.tolist()
    # also treat low-unique numeric as categorical
    for c in numeric_cols.copy():
        if df_train[c].nunique() / n_rows < categorical_threshold and df_train[c].nunique() < 50:
            cat_cols.append(c)
            numeric_cols.remove(c)

    print("Numeric cols:", len(numeric_cols), "Cat cols:", len(cat_cols))
    # numeric pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # categorical pipeline: target encoding for high-card, one-hot for small-card
    small_card = [c for c in cat_cols if df_train[c].nunique() <= 10]
    large_card = [c for c in cat_cols if df_train[c].nunique() > 10]
    print("small_card:", len(small_card), "large_card:", len(large_card))

    cat_transformer_small = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    cat_transformer_large = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        # replace high-card with target encoder later in model loop if needed
    ])
    # Build ColumnTransformer, but we'll handle target encoding separately as it needs y
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat_small', cat_transformer_small, small_card)
    ], remainder='drop', sparse_threshold=0)

    return preprocessor, numeric_cols, small_card, large_card

# Training loop with cross-validation
def train_and_predict(df_train, df_test=None, sample_sub=None):
    infer_columns(df_train)
    y       = df_train[TARGET_COL]
    task    = detect_task(y)
    print("Detected task:", task)

    quick_eda(df_train.drop(columns=[TARGET_COL]))
    preprocessor, numeric_cols, small_card, large_card = build_pipelines(df_train.drop(columns=[TARGET_COL]))
    # Prepare folds
    if task == 'classification' and y.nunique() > 1:
        folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    else:
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    X       = df_train.drop(columns=[TARGET_COL])
    X_test  = df_test.copy() if df_test is not None else None

    # placeholder arrays
    oof_preds = np.zeros(len(X))
    if X_test is not None:
        test_preds = np.zeros((len(X_test),))

    feature_importances = pd.DataFrame()

    # We'll use LightGBM as primary baseline, fallback to XGBoost if desired
    for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y if task=='classification' else None)):
        print(f"FOLD {fold+1}")
        X_tr, X_val = X.iloc[tr_idx].reset_index(drop=True), X.iloc[val_idx].reset_index(drop=True)
        y_tr, y_val = y.iloc[tr_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)

        # Fit preprocessor on train
        preprocessor.fit(X_tr)
        X_tr_proc   = preprocessor.transform(X_tr)
        X_val_proc  = preprocessor.transform(X_val)
        X_test_proc = preprocessor.transform(X_test) if X_test is not None else None

        # handle large-card categorical with target encoding
        if large_card:
            # we will use target encoder trained on X_tr
            te = ce.TargetEncoder(cols=large_card, smoothing=0.3)
            te.fit(X_tr[large_card], y_tr)
            tr_te   = te.transform(X_tr[large_card])
            val_te  = te.transform(X_val[large_card])
            if X_test is not None:
                test_te = te.transform(X_test[large_card])
            # concatenate with processed arrays
            # careful: preprocessor output is numpy array; need to concatenate columns
            X_tr_proc   = np.hstack([X_tr_proc, tr_te.values])
            X_val_proc  = np.hstack([X_val_proc, val_te.values])
            if X_test is not None:
                X_test_proc = np.hstack([X_test_proc, test_te.values])

        callbacks = [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]

        # LightGBM dataset
        if task == 'classification':
            lgb_train   = lgb.Dataset(X_tr_proc, label=y_tr)
            lgb_val     = lgb.Dataset(X_val_proc, label=y_val, reference=lgb_train)
            params      = {
                'objective': 'binary' if y.nunique() == 2 else 'multiclass',
                'metric': 'binary_logloss' if y.nunique() == 2 else 'multi_logloss',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'seed': RANDOM_STATE,
                'n_jobs': -1
            }

            if y.nunique() > 2:
                params['num_class'] = y.nunique()

            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=[lgb_train, lgb_val],
                callbacks=callbacks
            )
            val_pred = model.predict(X_val_proc, num_iteration=model.best_iteration)
            if X_test is not None:
                test_pred = model.predict(X_test_proc, num_iteration=model.best_iteration)
        else:
            # regression
            lgb_train   = lgb.Dataset(X_tr_proc, label=y_tr)
            lgb_val     = lgb.Dataset(X_val_proc, label=y_val, reference=lgb_train)
            params      = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'seed': RANDOM_STATE,
                'n_jobs': -1
            }

            model       = lgb.train(
                params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=[lgb_train, lgb_val],
                callbacks=callbacks
            )
            val_pred    = model.predict(X_val_proc, num_iteration=model.best_iteration)
            if X_test is not None:
                test_pred = model.predict(X_test_proc, num_iteration=model.best_iteration)

        # scoring
        if task == 'classification':
            if y.nunique() == 2:
                score = roc_auc_score(y_val, val_pred)
                print("Fold ROC AUC:", score)
            else:
                try:
                    # multi-class: use log loss and macro roc_auc if possible
                    score = log_loss(y_val, val_pred)
                    print("Fold log loss:", score)
                except Exception as e:
                    print("Fold score could not be computed:", e)
        else:
            rmse    = np.sqrt(mean_squared_error(y_val, val_pred))
            mae     = mean_absolute_error(y_val, val_pred)
            print(f"Fold RMSE: {rmse:.5f}, MAE: {mae:.5f}")

        oof_preds[val_idx] = val_pred if val_pred.ndim==1 else np.argmax(val_pred, axis=1)
        if X_test is not None:
            test_preds += test_pred if test_pred.ndim==1 else np.argmax(test_pred,axis=1)
        # save feature importances if available
        try:
            fi = pd.DataFrame({
                'feature': list(range(X_tr_proc.shape[1])),
                'importance': model.feature_importance()
            })
            fi['fold'] = fold+1
            feature_importances = pd.concat([feature_importances, fi], ignore_index=True)
        except:
            pass

    # Average test predictions
    if X_test is not None:
        test_preds = test_preds / N_FOLDS

    # OOF score
    if task == 'classification':
        if y.nunique() == 2:
            overall_score = roc_auc_score(y, oof_preds)
            print("OOF ROC AUC:", overall_score)
        else:
            try:
                overall_score = log_loss(y, oof_preds)
                print("OOF log loss:", overall_score)
            except:
                print("OOF score calculation failed for multi-class.")
    else:
        overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        print("OOF RMSE:", overall_rmse)

    # Save a model and feature importances
    joblib.dump(model, "baseline_lgb_model.joblib")
    if not feature_importances.empty:
        feature_importances.to_csv("feature_importances.csv", index=False)

    # Create submission
    if X_test is not None and sample_sub is not None:
        submission = sample_sub.copy()
        # heuristics: find prediction column in sample_sub
        pred_cols = [c for c in submission.columns if c not in (ID_COL,)]
        if len(pred_cols) == 1:
            submission[pred_cols[0]] = test_preds
        elif len(pred_cols) >= 1:
            # if multi-class, ensure shape matches
            if isinstance(test_preds, np.ndarray) and test_preds.ndim==2:
                for i, c in enumerate(pred_cols):
                    submission[c] = test_preds[:,i]
            else:
                submission[pred_cols[0]] = test_preds
        else:
            # fallback: create 'prediction' column
            submission['prediction'] = test_preds
        submission.to_csv("submission_baseline.csv", index=False)
        print("Wrote submission_baseline.csv")

    return model, oof_preds, (test_preds if X_test is not None else None), feature_importances

# --- Main execution ---
if __name__ == "__main__":
    # Run loader
    _df_train, _df_test, _sample_sub = load_data()

    # If target unknown, set it explicitly here. Otherwise, the script will attempt to infer.
    # Example: TARGET_COL = 'target'
    _model, _oof, _test_preds, _feature_importances = train_and_predict(_df_train, _df_test, _sample_sub)
