import os
import math
import time
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_MIN, TARGET_MAX = 0.0, 10.0
ID_COL_DEFAULT = "ID"
DAY_COL_DEFAULT = "Day"
LABEL_COL_DEFAULT = "Ystar_next_day"
BASE_COMPOSITE_COL = "U_t"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)

def load_ml_ready(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    print(f"[OK] Loaded dataset: {path} | shape={df.shape}")
    return df

def gpu_available() -> bool:
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return True
    except Exception:
        pass
    return os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ["", "-1", None]

def _safe_has_xgb() -> bool:
    try:
        import xgboost
        return True
    except Exception:
        return False

def _safe_has_catboost() -> bool:
    try:
        import catboost
        return True
    except Exception:
        return False

def ensure_next_day_label(df: pd.DataFrame,
                          id_col: str = ID_COL_DEFAULT,
                          day_col: str = DAY_COL_DEFAULT,
                          label_col: str = LABEL_COL_DEFAULT,
                          base_col: str = BASE_COMPOSITE_COL) -> pd.DataFrame:

    out = df.copy()
    if label_col not in out.columns:
        if base_col not in out.columns:
            raise ValueError(
                f"Cannot create {label_col}: base composite '{base_col}' not found."
            )
        out = out.sort_values([id_col, day_col])
        out[label_col] = out.groupby(id_col, group_keys=False)[base_col].shift(-1)
        print(f"[OK] Created label '{label_col}' from '{base_col}' (shift -1 within {id_col}).")
    before = len(out)
    out = out.dropna(subset=[label_col]).copy()
    after = len(out)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with missing next-day label.")
    return out

def split_by_id_time(df: pd.DataFrame,
                     id_col: str = ID_COL_DEFAULT,
                     day_col: str = DAY_COL_DEFAULT,
                     test_size: float = 0.2,
                     val_size: float = 0.2,
                     seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)
    ids = df[id_col].dropna().unique()
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(test_size * n)))
    n_val  = max(1, int(round(val_size * (n - n_test))))
    test_ids = set(ids[:n_test])
    val_ids  = set(ids[n_test:n_test + n_val])
    train_ids = set(ids[n_test + n_val:])
    train = df[df[id_col].isin(train_ids)].sort_values([id_col, day_col]).copy()
    val   = df[df[id_col].isin(val_ids)].sort_values([id_col, day_col]).copy()
    test  = df[df[id_col].isin(test_ids)].sort_values([id_col, day_col]).copy()
    print(f"[SPLIT] IDs -> train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"[SPLIT] Rows -> train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test

def build_feature_label_matrices(df: pd.DataFrame,
                                 id_col: str = ID_COL_DEFAULT,
                                 day_col: str = DAY_COL_DEFAULT,
                                 label_col: str = LABEL_COL_DEFAULT,
                                 feature_cols: Optional[List[str]] = None
                                 ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:

    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop = {id_col, day_col, label_col}
        feature_cols = [c for c in numeric_cols if c not in drop]
    X = df[feature_cols].copy()
    y = df[label_col].to_numpy().astype(float)
    return X, y, feature_cols

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def save_model(pipeline: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[OK] Saved model -> {path}")

def save_cv_results(model_name: str,
                    cv_results_: Dict[str, Any],
                    out_dir: str) -> str:

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(cv_results_)
    path = os.path.join(out_dir, f"{model_name}_cv_results.csv")
    df.to_csv(path, index=False)
    print(f"[OK] Saved CV grid -> {path}")
    return path

def make_pipeline_and_grid(model_name: str, use_gpu: bool):
    from models import (
        LassoModel, ElasticNetModel, RandomForestModel,
        XGBoostModel, SVRModel, KNNRegressorModel, CatBoostRegressorModel
    )
    m = model_name.lower()

    if m == "lasso":
        pipe = LassoModel().sklearn_pipeline()
        grid = {
            "model__alpha": np.logspace(-3, 1, 7),
        }
        return pipe, grid

    if m == "elasticnet":
        pipe = ElasticNetModel().sklearn_pipeline()
        grid = {
            "model__alpha": np.logspace(-3, 1, 7),
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        return pipe, grid

    if m == "rf":
        pipe = RandomForestModel().sklearn_pipeline()
        grid = {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 6, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.6, 0.9],
        }
        return pipe, grid

    if m == "svr":
        pipe = SVRModel().sklearn_pipeline()
        grid = {
            "model__kernel": ["rbf"],
            "model__C": [0.5, 1.0, 2.0, 5.0],
            "model__epsilon": [0.05, 0.1, 0.2],
            "model__gamma": ["scale", "auto"],
        }
        return pipe, grid

    if m == "knn":
        pipe = KNNRegressorModel().sklearn_pipeline()
        grid = {
            "model__n_neighbors": [3, 5, 9, 15],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
        return pipe, grid

    if m == "xgb":
        if not _safe_has_xgb():
            raise ImportError("xgboost not installed.")
        # Base estimator (GPU if available)
        tree_method = "gpu_hist" if use_gpu else "hist"
        from xgboost import XGBRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        est = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method=tree_method,
            random_state=42,
            n_jobs=-1,
        )
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", est),
        ])
        grid = {
            "model__n_estimators": [300, 600, 900],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
        }
        return pipe, grid

    if m == "catboost":
        if not _safe_has_catboost():
            raise ImportError("catboost not installed.")
        task_type = "GPU" if use_gpu else "CPU"
        from catboost import CatBoostRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        est = CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.9,
            random_seed=42,
            task_type=task_type,
            loss_function="RMSE",
            verbose=False,
        )
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", est),
        ])
        grid = {
            "model__iterations": [400, 600, 900],
            "model__depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__l2_leaf_reg": [1.0, 3.0, 7.0],
            "model__subsample": [0.8, 1.0],
        }
        return pipe, grid

    raise ValueError(f"Unknown model_name: {model_name}")

def timeit(fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = fn(*args, **kwargs)
        dt = time.time() - t0
        print(f"[TIME] {fn.__name__}: {dt:.2f}s")
        return res
    return wrapper

def run_grid_search(model_name: str,
                    X: pd.DataFrame,
                    y: np.ndarray,
                    cv_splits: int = 5,
                    out_dir: str = "../Data/grid_results",
                    refit_metric: str = "r2") -> Dict[str, Any]:

    os.makedirs(out_dir, exist_ok=True)
    use_gpu = gpu_available()
    est, grid = make_pipeline_and_grid(model_name, use_gpu)

    scoring = {
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    cv = KFold(n_splits=cv_splits, shuffle=False)
    gs = GridSearchCV(
        estimator=est,
        param_grid=grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )
    gs.fit(X, y)
    res_df = pd.DataFrame(gs.cv_results_)
    res_csv = os.path.join(out_dir, f"{model_name}_cv_results.csv")
    res_df.to_csv(res_csv, index=False)
    best_pkl = os.path.join(out_dir, f"{model_name}_best.pkl")
    joblib.dump(gs.best_estimator_, best_pkl)
    bi = gs.best_index_
    mean_mae = -res_df.loc[bi, "mean_test_mae"]
    std_mae  =  res_df.loc[bi, "std_test_mae"]
    mean_r2  =  res_df.loc[bi, "mean_test_r2"]
    std_r2   =  res_df.loc[bi, "std_test_r2"]
    summary = {
        "model": model_name,
        "use_gpu": use_gpu,
        "best_params": gs.best_params_,
        "best_model_path": best_pkl,
        "cv_results_csv": res_csv,
        "MAE_mean": round(float(mean_mae), 4),
        "MAE_sd": round(float(std_mae), 4),
        "R2_mean": round(float(mean_r2), 4),
        "R2_sd": round(float(std_r2), 4),
    }
    print(f"[{model_name}] best R²={summary['R2_mean']:.3f} ± {summary['R2_sd']:.3f} | "
          f"MAE={summary['MAE_mean']:.3f} ± {summary['MAE_sd']:.3f} | GPU={use_gpu}")
    return summary