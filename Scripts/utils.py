import os
import math
import time
import itertools
import joblib
import shap
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager
from sympy.benchmarks.bench_discrete_log import data_set_1
from tqdm import tqdm
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False
    CatBoostRegressor = None

try:
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.svm import SVR as cuSVR
    from cuml.neighbors import KNeighborsRegressor as cuKNN
    from cuml.linear_model import Lasso as cuLasso, ElasticNet as cuElasticNet
    _HAS_CUML = True
except Exception:
    _HAS_CUML = False
    cuRF = cuSVR = cuKNN = cuLasso = cuElasticNet = None

TARGET_MIN = 0.0
TARGET_MAX = 10.0
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
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return False
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            return True
    except Exception:
        pass
    try:
        from numba import cuda as nb_cuda
        if nb_cuda.is_available():
            return True
    except Exception:
        pass
    try:
        import torch
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False

def cuml_available() -> bool:
    return _HAS_CUML

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

def ensure_next_day_label(df: pd.DataFrame, id_col: str = ID_COL_DEFAULT, day_col: str = DAY_COL_DEFAULT, label_col: str = LABEL_COL_DEFAULT, base_col: str = BASE_COMPOSITE_COL) -> pd.DataFrame:
    out = df.copy()
    if label_col not in out.columns:
        if base_col not in out.columns:
            raise ValueError(f"Cannot create {label_col}: base composite '{base_col}' not found.")
        out = out.sort_values([id_col, day_col])
        out[label_col] = out.groupby(id_col, group_keys=False)[base_col].shift(-1)
        print(f"[OK] Created label '{label_col}' from '{base_col}'.")
    before = len(out)
    out = out.dropna(subset=[label_col]).copy()
    after = len(out)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with missing {label_col}.")
    return out

def ensure_horizon_label(df: pd.DataFrame,
                         h: int,
                         id_col: str = ID_COL_DEFAULT,
                         day_col: str = DAY_COL_DEFAULT,
                         base_col: str = BASE_COMPOSITE_COL,
                         use_delta: bool = True) -> Tuple[pd.DataFrame, str]:
    out = df.sort_values([id_col, day_col]).copy()
    abs_col = f"Y_next_h{h}"
    dy_col = f"dY_next_h{h}"
    if abs_col not in out.columns:
        if base_col not in out.columns:
            raise ValueError(f"Missing {base_col} for horizon {h}")
        out[abs_col] = out.groupby(id_col, group_keys=False)[base_col].shift(-h)
    if use_delta:
        if dy_col not in out.columns:
            out[dy_col] = out[abs_col] - out[base_col]
        label_col = dy_col
    else:
        label_col = abs_col
    before = len(out)
    out = out.dropna(subset=[label_col]).copy()
    after = len(out)
    if before != after:
        print(f"[H={h}] Dropped {before - after} rows with missing {label_col}.")
    return out, label_col

def split_by_id_time(df: pd.DataFrame, id_col: str = ID_COL_DEFAULT, day_col: str = DAY_COL_DEFAULT, test_size: float = 0.2, val_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = df[id_col].dropna().unique()
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(test_size * n)))
    n_val = max(1, int(round(val_size * (n - n_test))))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + n_val])
    train_ids = set(ids[n_test + n_val:])
    train = df[df[id_col].isin(train_ids)].sort_values([id_col, day_col]).copy()
    val = df[df[id_col].isin(val_ids)].sort_values([id_col, day_col]).copy()
    test = df[df[id_col].isin(test_ids)].sort_values([id_col, day_col]).copy()
    print(f"[SPLIT] IDs train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"[SPLIT] Rows train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test

def build_feature_label_matrices(df: pd.DataFrame,
                                 id_col: str = ID_COL_DEFAULT,
                                 day_col: str = DAY_COL_DEFAULT,
                                 label_col: str = LABEL_COL_DEFAULT,
                                 feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop = {id_col, day_col, label_col}
        leak_prefixes = ("Y_next_h", "dY_next_h")
        leak_exact = {"Ystar_next_day", "dY_next"}
        feature_cols = []
        for c in numeric_cols:
            if c in drop:
                continue
            if any(c.startswith(p) for p in leak_prefixes):
                continue
            if c in leak_exact:
                continue
            feature_cols.append(c)
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    y = df[label_col].to_numpy().astype(float)
    return X, y, feature_cols

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def metric_ci_from_mean_sd(mean: float, sd: float, n: int, level: float = 0.95) -> Dict[str, float]:
    if n <= 1 or not np.isfinite(sd):
        return {
            "mean": float(mean),
            "sd": float(sd),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_half_width": float("nan"),
        }
    z = 1.96 if abs(level - 0.95) < 1e-6 else 1.96
    half = z * sd / np.sqrt(n)
    return {
        "mean": float(mean),
        "sd": float(sd),
        "ci_low": float(mean - half),
        "ci_high": float(mean + half),
        "ci_half_width": float(half),
    }

def save_model(pipeline: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[OK] Saved model -> {path}")

def save_cv_results(model_name: str, cv_results_: Dict[str, Any], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(cv_results_)
    path = os.path.join(out_dir, f"{model_name}_cv_results.csv")
    df.to_csv(path, index=False)
    print(f"[OK] Saved CV grid -> {path}")
    return path

class CuMLCompatRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, preprocessor: SkPipeline, model):
        self.preprocessor = preprocessor
        self.model = model
        self._is_fitted = False
    def get_params(self, deep=True):
        params = {"preprocessor": self.preprocessor, "model": self.model}
        if deep:
            for name, val in self.preprocessor.get_params(deep=True).items():
                params[f"preprocessor__{name}"] = val
            try:
                for name, val in self.model.get_params(deep=True).items():
                    params[f"model__{name}"] = val
            except Exception:
                pass
        return params
    def set_params(self, **params):
        pre_params = {k.split("__",1)[1]: v for k,v in params.items() if k.startswith("preprocessor__")}
        mdl_params = {k.split("__",1)[1]: v for k,v in params.items() if k.startswith("model__")}
        if "preprocessor" in params:
            self.preprocessor = params["preprocessor"]
        if "model" in params:
            self.model = params["model"]
        if pre_params:
            self.preprocessor.set_params(**pre_params)
        if mdl_params:
            try:
                self.model.set_params(**mdl_params)
            except Exception:
                pass
        return self
    def fit(self, X, y):
        self.preprocessor.fit(X, y)
        Xp = self.preprocessor.transform(X)
        self.model.fit(Xp, y)
        self._is_fitted = True
        return self
    def predict(self, X):
        Xp = self.preprocessor.transform(X)
        return self.model.predict(Xp)
    def __sklearn_is_fitted__(self):
        return self._is_fitted
    def __sklearn_tags__(self):
        return SimpleNamespace(
            estimator_type="regressor",
            input_tags=SimpleNamespace(pairwise=False),
            X_types=["2darray"],
            allow_nan=True,
            non_deterministic=True,
            poor_score=False,
            requires_positive_X=False,
            requires_y=True,
        )

def make_pipeline_and_grid(model_name: str, use_gpu: bool):
    from models import LassoModel, ElasticNetModel, RandomForestModel, XGBoostModel, SVRModel, KNNRegressorModel, CatBoostRegressorModel
    m = model_name.lower()
    if m == "lasso":
        if use_gpu and cuml_available():
            est = cuLasso()
            pre = SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            pipe = CuMLCompatRegressor(pre, est)
            grid = {"model__alpha": np.logspace(-5, 1, 9).tolist(), "model__fit_intercept": [True]}
            return pipe, grid
        pipe = LassoModel().sklearn_pipeline()
        grid = {"model__alpha": np.logspace(-5, 1, 9).tolist()}
        return pipe, grid
    if m == "elasticnet":
        if use_gpu and cuml_available():
            est = cuElasticNet()
            pre = SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            pipe = CuMLCompatRegressor(pre, est)
            grid = {"model__alpha": np.logspace(-5, 1, 9).tolist(), "model__l1_ratio": [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]}
            return pipe, grid
        pipe = ElasticNetModel().sklearn_pipeline()
        grid = {"model__alpha": np.logspace(-5, 1, 9).tolist(), "model__l1_ratio": [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]}
        return pipe, grid
    if m == "rf":
        if use_gpu and cuml_available():
            est = cuRF(random_state=42)
            pre = SkPipeline([("imputer", SimpleImputer(strategy="median"))])
            pipe = CuMLCompatRegressor(pre, est)
            grid = {"model__n_estimators": [250, 500, 1000], "model__max_depth": [6, 12, 18], "model__max_features": [0.5, 0.7, 1.0], "model__min_samples_leaf": [1, 2, 4]}
            return pipe, grid
        pipe = RandomForestModel().sklearn_pipeline()
        grid = {"model__n_estimators": [250, 500, 1000], "model__max_depth": [None, 10, 18], "model__min_samples_leaf": [1, 2, 4], "model__min_samples_split": [2, 8, 16], "model__max_features": ["sqrt", "log2", 0.7]}
        return pipe, grid
    if m == "svr":
        if use_gpu and cuml_available():
            est = cuSVR()
            pre = SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            pipe = CuMLCompatRegressor(pre, est)
            grid = {"model__kernel": ["rbf", "linear"], "model__C": [0.5, 1.0, 4.0], "model__epsilon": [0.05, 0.2, 0.4], "model__gamma": ["scale", 0.1]}
            return pipe, grid
        pipe = SVRModel().sklearn_pipeline()
        grid = {"model__kernel": ["rbf", "poly"], "model__C": [0.5, 1.0, 4.0], "model__epsilon": [0.05, 0.2, 0.4], "model__gamma": ["scale", "auto", 0.1], "model__degree": [2, 3], "model__coef0": [0.0, 0.5]}
        return pipe, grid
    if m == "knn":
        if use_gpu and cuml_available():
            est = cuKNN()
            pre = SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            pipe = CuMLCompatRegressor(pre, est)
            grid = {"model__n_neighbors": [5, 9, 15, 31], "model__weights": ["uniform", "distance"], "model__p": [1, 2]}
            return pipe, grid
        from models import KNNRegressorModel
        pipe = KNNRegressorModel().sklearn_pipeline()
        grid = {"model__n_neighbors": [5, 9, 15, 31], "model__weights": ["uniform", "distance"], "model__p": [1, 2], "model__metric": ["minkowski"]}
        return pipe, grid
    if m == "xgb":
        if not _safe_has_xgb():
            raise ImportError("xgboost not installed.")
        est = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, tree_method="hist", predictor="auto")
        pipe = SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", est),
        ])
        grid = {
            "model__n_estimators": [250, 500, 1000],
            "model__max_depth": [3, 7, 11],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_alpha": [0.0, 0.5, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0],
        }
        return pipe, grid
    if m == "catboost":
        if not _safe_has_catboost():
            raise ImportError("catboost not installed.")
        task_type = "GPU" if use_gpu else "CPU"
        est = CatBoostRegressor(loss_function="RMSE", random_seed=42, task_type=task_type, verbose=False)
        pipe = SkPipeline([("imputer", SimpleImputer(strategy="median")), ("model", est)])
        grid = {
            "model__iterations": [250, 500, 1000],
            "model__depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__l2_leaf_reg": [1.0, 3.0, 7.0],
            "model__bagging_temperature": [0, 1, 3],
            "model__random_strength": [1, 5, 10],
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

def run_nn_grid(X: pd.DataFrame, y: np.ndarray, cv_splits: int = 5, out_dir: str = "../Data/grid_results", refit_metric: str = "r2", max_combos: int = 80) -> Dict[str, Any]:
    from models import TorchMLPModel
    os.makedirs(out_dir, exist_ok=True)
    param_grid = {
        "hidden_sizes": [(256,128), (128,64), (128,64,32), (64,32)],
        "dropout": [0.0, 0.2, 0.4],
        "use_batchnorm": [False, True],
        "bn_momentum": [0.1, 0.01],
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [0.0, 1e-6, 1e-5],
        "batch_size": [32, 64, 128],
        "epochs": [50],
        "patience": [10],
        "scheduler_type": ["none", "step", "cosine"],
        "scheduler_step": [10, 20],
        "scheduler_gamma": [0.8, 0.5],
        "random_state": [42],
    }
    keys, vals = zip(*param_grid.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    if len(all_combos) > max_combos:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_combos), size=max_combos, replace=False)
        combos = [all_combos[i] for i in idx]
        print(f"[mlp] sampling {len(combos)} of {len(all_combos)} configs")
    else:
        combos = all_combos
        print(f"[mlp] trying {len(combos)} configs")
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    fold_rows = []
    summary_rows = []
    best_score = -np.inf
    best_cfg = None
    for cfg_id, cfg in enumerate(combos):
        scores = []
        for fold_idx, (tr, va) in enumerate(cv.split(X)):
            Xt, Xv = X.iloc[tr], X.iloc[va]
            yt, yv = y[tr], y[va]
            model = TorchMLPModel.from_config(cfg)
            model.fit(Xt, yt)
            preds = model.predict(Xv)
            if refit_metric == "r2":
                score = r2_score(yv, preds)
            else:
                score = -mean_absolute_error(yv, preds)
            scores.append(score)
            row = {"config_id": cfg_id, "fold": fold_idx, "score": float(score)}
            row.update(cfg)
            fold_rows.append(row)
        mean_score = float(np.mean(scores))
        srow = {"config_id": cfg_id, "mean_score": mean_score}
        srow.update(cfg)
        summary_rows.append(srow)
        if mean_score > best_score:
            best_score = mean_score
            best_cfg = cfg
    folds_df = pd.DataFrame(fold_rows)
    folds_csv = os.path.join(out_dir, "mlp_cv_folds_detailed.csv")
    folds_df.to_csv(folds_csv, index=False)
    res_df = pd.DataFrame(summary_rows)
    res_csv = os.path.join(out_dir, "mlp_cv_results.csv")
    res_df.to_csv(res_csv, index=False)
    best_model = TorchMLPModel.from_config(best_cfg).fit(X, y)
    best_pkl = os.path.join(out_dir, "mlp_best.pkl")
    best_model.save(best_pkl)
    epoch_csv = None
    if hasattr(best_model, "epoch_logs_") and best_model.epoch_logs_:
        epoch_df = pd.DataFrame(best_model.epoch_logs_)
        epoch_csv = os.path.join(out_dir, "mlp_best_epoch_logs.csv")
        epoch_df.to_csv(epoch_csv, index=False)
    summary = {
        "model": "mlp",
        "use_gpu": gpu_available(),
        "best_params": best_cfg,
        "best_model_path": best_pkl,
        "cv_results_csv": res_csv,
        "cv_folds_csv": folds_csv,
        "epoch_logs_csv": epoch_csv,
        "score": round(best_score, 4),
    }
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(out_dir, "mlp_cv_summary_with_ci.csv")
    summary_df.to_csv(summary_csv, index=False)
    summary["summary_csv"] = summary_csv
    print(f"[mlp] best {refit_metric}={summary['score']:.3f} | GPU={summary['use_gpu']}")
    return summary

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

def run_grid_search(model_name: str, X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray] = None, cv_splits: int = 5, out_dir: str = "../Data/grid_results", refit_metric: str = "r2") -> Dict[str, Any]:
    if model_name.lower() in ("mlp", "nn", "torchmlp"):
        return run_nn_grid(X, y, cv_splits=cv_splits, out_dir=out_dir, refit_metric=refit_metric)
    os.makedirs(out_dir, exist_ok=True)
    if model_name.lower() in ("rf", "svr", "knn", "lasso", "elasticnet"):
        use_gpu = gpu_available() and cuml_available()
    elif model_name.lower() in ("xgb", "catboost"):
        use_gpu = gpu_available()
    else:
        use_gpu = False
    est, grid = make_pipeline_and_grid(model_name, use_gpu)
    scoring = {"mae": "neg_mean_absolute_error", "r2": "r2"}
    if groups is not None:
        cv = GroupKFold(n_splits=cv_splits)
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    n_cand = len(ParameterGrid(grid))
    total = n_cand * cv_splits
    gs = GridSearchCV(
        estimator=est,
        param_grid=grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
        error_score="raise",
    )
    with tqdm_joblib(tqdm(total=total, desc=f"{model_name} CV", unit="fit")):
        if groups is not None:
            gs.fit(X, y, groups=groups)
        else:
            gs.fit(X, y)
    res_df = pd.DataFrame(gs.cv_results_)
    res_csv = os.path.join(out_dir, f"{model_name}_cv_results.csv")
    res_df.to_csv(res_csv, index=False)
    best_pkl = os.path.join(out_dir, f"{model_name}_best.pkl")
    joblib.dump(gs.best_estimator_, best_pkl)
    bi = gs.best_index_
    mean_mae = -float(res_df.loc[bi, "mean_test_mae"])
    std_mae = float(res_df.loc[bi, "std_test_mae"])
    mean_r2 = float(res_df.loc[bi, "mean_test_r2"])
    std_r2 = float(res_df.loc[bi, "std_test_r2"])
    mae_ci = metric_ci_from_mean_sd(mean_mae, std_mae, cv_splits, level=0.95)
    r2_ci = metric_ci_from_mean_sd(mean_r2, std_r2, cv_splits, level=0.95)
    summary = {
        "model": model_name,
        "use_gpu": use_gpu,
        "best_params": gs.best_params_,
        "best_model_path": best_pkl,
        "cv_results_csv": res_csv,
        "MAE_mean": round(mae_ci["mean"], 4),
        "MAE_sd": round(mae_ci["sd"], 4),
        "MAE_ci_low": round(mae_ci["ci_low"], 4),
        "MAE_ci_high": round(mae_ci["ci_high"], 4),
        "MAE_ci_half_width": round(mae_ci["ci_half_width"], 4),
        "R2_mean": round(r2_ci["mean"], 4),
        "R2_sd": round(r2_ci["sd"], 4),
        "R2_ci_low": round(r2_ci["ci_low"], 4),
        "R2_ci_high": round(r2_ci["ci_high"], 4),
        "R2_ci_half_width": round(r2_ci["ci_half_width"], 4),
    }
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(out_dir, f"{model_name}_cv_summary_with_ci.csv")
    summary_df.to_csv(summary_csv, index=False)
    summary["summary_csv"] = summary_csv
    print(
        f"[{model_name}] best R2={summary['R2_mean']:.3f} "
        f"(95% CI {summary['R2_ci_low']:.3f}–{summary['R2_ci_high']:.3f}) "
        f"MAE={summary['MAE_mean']:.3f} "
        f"(95% CI {summary['MAE_ci_low']:.3f}–{summary['MAE_ci_high']:.3f}) "
        f"GPU={use_gpu}"
    )
    return summary

def run_all_models(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray] = None, models: List[str] = ("lasso","elasticnet","rf","svr","knn","xgb","catboost","mlp"), cv_splits: int = 5, out_dir: str = "../Data/grid_results", refit_metric: str = "r2") -> Dict[str, Dict[str, Any]]:
    results = {}
    for m in models:
        print("=" * 80)
        print(f"Running model: {m}")
        results[m] = run_grid_search(m, X, y, groups=groups, cv_splits=cv_splits, out_dir=out_dir, refit_metric=refit_metric)
    return results

def _bg_sample(Xp: np.ndarray, max_bg: int = 200, seed: int = 42) -> np.ndarray:
    if len(Xp) <= max_bg:
        return Xp
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xp), size=max_bg, replace=False)
    return Xp[idx]

def shap_mean_abs_importance(shap_values: np.ndarray, feature_names) -> pd.Series:
    vals = np.array(shap_values)
    if vals.ndim == 3:
        vals = vals.mean(axis=0)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    n_feat = vals.shape[1]
    names = list(feature_names)
    if len(names) > n_feat:
        names = names[:n_feat]
    elif len(names) < n_feat:
        extra = [f"feat_{i}" for i in range(len(names), n_feat)]
        names = names + extra
    return pd.Series(np.mean(np.abs(vals), axis=0), index=names).sort_values(ascending=False)

def shap_importance_for_pipeline(pipe, X: pd.DataFrame, top_n: int = 20) -> pd.Series:
    X = X.apply(pd.to_numeric, errors="coerce")
    pre = None
    model = None
    feat_names = list(X.columns)
    if hasattr(pipe, "preprocessor") and hasattr(pipe, "model"):
        pre = pipe.preprocessor
        model = pipe.model
    elif isinstance(pipe, SkPipeline):
        pre = pipe[:-1]
        model = pipe[-1]
    else:
        model = pipe
    if pre is not None:
        Xp = pre.transform(X)
    else:
        Xp = X.values
    Xp = np.asarray(Xp, dtype=float)
    name = model.__class__.__name__.lower()
    if any(k in name for k in ["forest", "xgb", "catboost", "tree"]):
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feat_names)
            return imp.sort_values(ascending=False).head(top_n)
        else:
            raise RuntimeError("Tree-like model without feature_importances_.")
    if any(k in name for k in ["lasso", "elasticnet", "linear"]):
        bg = _bg_sample(Xp, 500)
        sv = shap.LinearExplainer(model, bg).shap_values(Xp)
    else:
        bg = _bg_sample(Xp, 200)
        sv = shap.KernelExplainer(model.predict, bg).shap_values(Xp, nsamples="auto")
    imp = shap_mean_abs_importance(np.array(sv), feat_names)
    return imp.head(top_n)

def predict_pipeline_compat(estimator, X):
    if hasattr(estimator, "preprocessor") and hasattr(estimator, "model"):
        try:
            return estimator.predict(X)
        except Exception:
            Xp = estimator.preprocessor.transform(X)
            return estimator.model.predict(Xp)
    if isinstance(estimator, SkPipeline):
        try:
            return estimator.predict(X)
        except Exception:
            pre = estimator[:-1]
            model = estimator[-1]
            Xp = pre.transform(X)
            return model.predict(Xp)
    return estimator.predict(X)

def find_treatment_episodes(df, id_col, day_col, treat_col):
    eps = []
    for pid in df[id_col].dropna().unique():
        g = df[df[id_col] == pid].sort_values(day_col)
        tdf = g[g[treat_col] == 1]
        days = []
        for d in tdf[day_col].tolist():
            if pd.isna(d):
                continue
            days.append(int(d))
        if len(days) == 0:
            continue
        start = days[0]
        prev = days[0]
        for d in days[1:]:
            if d == prev + 1:
                prev = d
            else:
                eps.append((pid, start, prev))
                start = d
                prev = d
        eps.append((pid, start, prev))
    return eps

def build_treated_events(df, L, H, id_col, day_col, treat_col, outcome_col):
    rows = []
    eps = find_treatment_episodes(df, id_col, day_col, treat_col)
    for pid, start, end in eps:
        g = df[df[id_col] == pid].sort_values(day_col)
        start = int(start)
        end = int(end)
        pre = list(range(start - L, start))
        course = list(range(start, end + 1))
        post = list(range(end + 1, end + H + 1))
        if pre[0] < g[day_col].min() or post[-1] > g[day_col].max():
            continue
        win_pre = g[g[day_col].isin(pre)]
        win_course = g[g[day_col].isin(course)]
        win_post = g[g[day_col].isin(post)]
        if len(win_pre) != L or len(win_course) != len(course) or len(win_post) != H:
            continue
        if win_pre[treat_col].sum() != 0:
            continue
        if (win_course[treat_col] != 1).any():
            continue
        if win_pre[outcome_col].isna().any() or win_course[outcome_col].isna().any() or win_post[outcome_col].isna().any():
            continue
        baseline = float(win_course[outcome_col].mean())
        future = float(win_post[outcome_col].mean())
        y_star = future - baseline
        centre_row = win_course.iloc[0].to_dict()
        centre_row["A_star"] = 1
        centre_row["Y_star"] = float(y_star)
        centre_row["centre_day"] = int(start)
        centre_row["episode_start"] = int(start)
        centre_row["episode_end"] = int(end)
        centre_row["episode_len"] = int(end - start + 1)
        rows.append(centre_row)
    return pd.DataFrame(rows)

def build_control_events(df, L, H, id_col, day_col, treat_col, outcome_col, max_per_id=None):
    rows = []
    for pid in df[id_col].dropna().unique():
        g = df[df[id_col] == pid].sort_values(day_col)
        days = []
        for d in g[day_col].tolist():
            if pd.isna(d):
                continue
            days.append(int(d))
        cand = []
        for c in days:
            s = g[g[day_col] == c]
            if len(s) == 0:
                continue
            row_c = s.iloc[0]
            if row_c[treat_col] != 0:
                continue
            pre = list(range(int(c) - L, int(c)))
            post = list(range(int(c) + 1, int(c) + H + 1))
            if pre[0] < g[day_col].min() or post[-1] > g[day_col].max():
                continue
            all_days = pre + [int(c)] + post
            win = g[g[day_col].isin(all_days)]
            if len(win) != len(all_days):
                continue
            if win[treat_col].sum() != 0:
                continue
            if win[outcome_col].isna().any():
                continue
            win_post = win[win[day_col].isin(post)]
            centre_val = float(row_c[outcome_col])
            y_star = float(win_post[outcome_col].mean() - centre_val)
            row = row_c.to_dict()
            row["A_star"] = 0
            row["Y_star"] = y_star
            row["centre_day"] = int(c)
            row["episode_start"] = int(c)
            row["episode_end"] = int(c)
            row["episode_len"] = 1
            cand.append(row)
        if max_per_id is not None and len(cand) > max_per_id:
            cand = cand[:max_per_id]
        rows.extend(cand)
    return pd.DataFrame(rows)

def compute_event_weights_by_id(events_df, id_col):
    counts = events_df[id_col].value_counts()
    w = events_df[id_col].apply(lambda x: 1.0 / counts[x]).tolist()
    return np.array(w, dtype=float)
