import numpy as np
import pandas as pd
import joblib
import shap
import importlib
from tqdm import tqdm
from typing import Optional, Tuple, List, Any
from sklearn.linear_model import LassoCV, Lasso, ElasticNetCV, ElasticNet, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

_HAS_XGB = importlib.util.find_spec("xgboost") is not None
if _HAS_XGB:
    from xgboost import XGBRegressor as _XGBRegressor
else:
    _XGBRegressor = None

_HAS_TORCH = importlib.util.find_spec("torch") is not None
torch = None
nn = None
TensorDataset = None
DataLoader = None

_HAS_CATBOOST = importlib.util.find_spec("catboost") is not None
if _HAS_CATBOOST:
    from catboost import CatBoostRegressor
else:
    CatBoostRegressor = None

TARGET_MIN, TARGET_MAX = 0.0, 10.0

def _clip(yhat: np.ndarray, lo: float = TARGET_MIN, hi: float = TARGET_MAX) -> np.ndarray:
    return np.clip(yhat, lo, hi)

def _numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    return X.select_dtypes(include=[np.number]).copy()

def _bg_sample(Xp: np.ndarray, max_bg: int = 500, seed: int = 42) -> np.ndarray:
    if len(Xp) <= max_bg:
        return Xp
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xp), size=max_bg, replace=False)
    return Xp[idx]

def _mean_abs_shap(values: np.ndarray, feature_names: List[str]) -> pd.Series:
    vals = np.array(values)
    if vals.ndim == 3:
        vals = vals.mean(axis=0)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    return pd.Series(np.mean(np.abs(vals), axis=0), index=feature_names).sort_values(ascending=False)

class RidgeModel:
    def __init__(self, n_splits: int = 5, random_state: int = 42, max_iter: int = 10000):
        self.n_splits = n_splits
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = Ridge(max_iter=max_iter)
        self.medians: Optional[pd.Series] = None
        self.feat_cols: Optional[List[str]] = None
        self.fitted = False

    def prepare(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        X = _numeric_df(X).astype(np.float64)
        if self.feat_cols is not None:
            for c in self.feat_cols:
                if c not in X.columns:
                    X[c] = np.nan
            X = X[self.feat_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.medians is None:
            self.medians = X.median(numeric_only=True).fillna(0.0)
        X = X.fillna(self.medians).fillna(0.0)
        Xs = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        np.clip(Xs, -10, 10, out=Xs)
        return np.nan_to_num(Xs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xn = _numeric_df(X)
        keep = (Xn.notna().any(axis=0)) & (Xn.std(axis=0, skipna=True).fillna(0) > 0)
        Xn = Xn.loc[:, keep]
        self.feat_cols = Xn.columns.tolist()
        Xs = self.prepare(Xn, fit_scaler=True)
        self.model.fit(Xs, y)
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame, clip_to_bounds: bool = True) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        Xs = self.prepare(X, fit_scaler=False)
        yhat = self.model.predict(Xs)
        return _clip(yhat) if clip_to_bounds else yhat

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, label: str = "Ridge") -> Tuple[float, float]:
        preds = self.predict(X)
        mae, r2 = mean_absolute_error(y, preds), r2_score(y, preds)
        print(f"{label:<12} | MAE: {mae:.3f} | R²: {r2:.3f}")
        return mae, r2

    def coefficients(self) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        coefs = pd.Series(self.model.coef_, index=self.feat_cols)
        return coefs[coefs != 0].sort_values(key=np.abs, ascending=False)

    def shap_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        Xs = self.prepare(X, fit_scaler=False)
        bg = _bg_sample(Xs, 500)
        expl = shap.LinearExplainer(self.model, bg)
        sv = expl.shap_values(Xs)
        return _mean_abs_shap(sv, self.feat_cols).head(top_n)

    def sklearn_pipeline(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(max_iter=self.max_iter))
        ])

    def save(self, path: str):
        joblib.dump(
            {"scaler": self.scaler, "model": self.model, "medians": self.medians, "feat_cols": self.feat_cols},
            path
        )

    def load(self, path: str):
        obj = joblib.load(path)
        self.scaler = obj["scaler"]
        self.model = obj["model"]
        self.medians = obj["medians"]
        self.feat_cols = obj["feat_cols"]
        self.fitted = True
        return self

class LassoModel:
    def __init__(self, n_splits: int = 5, random_state: int = 42, max_iter: int = 10000):
        self.n_splits = n_splits
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = LassoCV(cv=n_splits, random_state=random_state, n_jobs=-1, max_iter=max_iter)
        self.medians: Optional[pd.Series] = None
        self.feat_cols: Optional[List[str]] = None
        self.fitted = False

    def _prepare(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        X = _numeric_df(X).astype(np.float64)
        if self.feat_cols is not None:
            for c in self.feat_cols:
                if c not in X.columns: X[c] = np.nan
            X = X[self.feat_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.medians is None: self.medians = X.median(numeric_only=True).fillna(0.0)
        X = X.fillna(self.medians).fillna(0.0)
        Xs = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        np.clip(Xs, -10, 10, out=Xs)
        return np.nan_to_num(Xs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xn = _numeric_df(X)
        keep = (Xn.notna().any(axis=0)) & (Xn.std(axis=0, skipna=True).fillna(0) > 0)
        Xn = Xn.loc[:, keep]
        self.feat_cols = Xn.columns.tolist()
        Xs = self._prepare(Xn, fit_scaler=True)
        self.model.fit(Xs, y)
        self.fitted = True
        print(f"[LassoModel] fitted | alpha={self.model.alpha_:.6f}")
        return self

    def predict(self, X: pd.DataFrame, clip_to_bounds: bool = True) -> np.ndarray:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xs = self._prepare(X, fit_scaler=False)
        yhat = self.model.predict(Xs)
        return _clip(yhat) if clip_to_bounds else yhat

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, label: str = "Lasso") -> Tuple[float, float]:
        preds = self.predict(X)
        mae, r2 = mean_absolute_error(y, preds), r2_score(y, preds)
        print(f"{label:<12} | MAE: {mae:.3f} | R²: {r2:.3f}")
        return mae, r2

    def coefficients(self) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        coefs = pd.Series(self.model.coef_, index=self.feat_cols)
        return coefs[coefs != 0].sort_values(key=np.abs, ascending=False)

    def shap_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xs = self._prepare(X, fit_scaler=False)
        bg = _bg_sample(Xs, 500)
        expl = shap.LinearExplainer(self.model, bg)
        sv = expl.shap_values(Xs)
        return _mean_abs_shap(sv, self.feat_cols).head(top_n)

    def sklearn_pipeline(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(random_state=self.random_state, max_iter=self.max_iter))
        ])

    def save(self, path: str):
        joblib.dump({"scaler": self.scaler, "model": self.model, "medians": self.medians, "feat_cols": self.feat_cols}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.scaler, self.model, self.medians, self.feat_cols = obj["scaler"], obj["model"], obj["medians"], obj["feat_cols"]
        self.fitted = True
        return self

class ElasticNetModel:
    def __init__(self, n_splits: int = 5, random_state: int = 42, l1_ratio_grid: Optional[List[float]] = None, max_iter: int = 10000):
        if l1_ratio_grid is None: l1_ratio_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.n_splits = n_splits
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = ElasticNetCV(cv=n_splits, random_state=random_state, l1_ratio=l1_ratio_grid, max_iter=max_iter, n_jobs=-1)
        self.medians: Optional[pd.Series] = None
        self.feat_cols: Optional[List[str]] = None
        self.fitted = False

    def _prepare(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        X = _numeric_df(X).astype(np.float64)
        if self.feat_cols is not None:
            for c in self.feat_cols:
                if c not in X.columns: X[c] = np.nan
            X = X[self.feat_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.medians is None: self.medians = X.median(numeric_only=True).fillna(0.0)
        X = X.fillna(self.medians).fillna(0.0)
        Xs = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        np.clip(Xs, -10, 10, out=Xs)
        return np.nan_to_num(Xs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xn = _numeric_df(X)
        keep = (Xn.notna().any(axis=0)) & (Xn.std(axis=0, skipna=True).fillna(0) > 0)
        Xn = Xn.loc[:, keep]
        self.feat_cols = Xn.columns.tolist()
        Xs = self._prepare(Xn, fit_scaler=True)
        self.model.fit(Xs, y)
        self.fitted = True
        print(f"[ElasticNetModel] fitted | alpha={self.model.alpha_:.6f} | l1_ratio={self.model.l1_ratio_:.3f}")
        return self

    def predict(self, X: pd.DataFrame, clip_to_bounds: bool = True) -> np.ndarray:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xs = self._prepare(X, fit_scaler=False)
        yhat = self.model.predict(Xs)
        return _clip(yhat) if clip_to_bounds else yhat

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, label: str = "ElasticNet") -> Tuple[float, float]:
        preds = self.predict(X)
        mae, r2 = mean_absolute_error(y, preds), r2_score(y, preds)
        print(f"{label:<12} | MAE: {mae:.3f} | R²: {r2:.3f}")
        return mae, r2

    def coefficients(self) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        coefs = pd.Series(self.model.coef_, index=self.feat_cols)
        return coefs[coefs != 0].sort_values(key=np.abs, ascending=False)

    def shap_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xs = self._prepare(X, fit_scaler=False)
        bg = _bg_sample(Xs, 500)
        expl = shap.LinearExplainer(self.model, bg)
        sv = expl.shap_values(Xs)
        return _mean_abs_shap(sv, self.feat_cols).head(top_n)

    def sklearn_pipeline(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(random_state=self.random_state, max_iter=self.max_iter))
        ])

    def save(self, path: str):
        joblib.dump({"scaler": self.scaler, "model": self.model, "medians": self.medians, "feat_cols": self.feat_cols}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.scaler, self.model, self.medians, self.feat_cols = obj["scaler"], obj["model"], obj["medians"], obj["feat_cols"]
        self.fitted = True
        return self

class RandomForestModel:
    def __init__(self, n_estimators: int = 300, max_depth: Optional[int] = None, min_samples_leaf: int = 2, max_features: str = "sqrt", random_state: int = 42, n_jobs: int = -1):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=random_state, n_jobs=n_jobs)
        self.model = RandomForestRegressor(**self.params)
        self.medians: Optional[pd.Series] = None
        self.feat_cols: Optional[List[str]] = None
        self.fitted = False

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _numeric_df(X)
        if self.feat_cols is not None:
            for c in self.feat_cols:
                if c not in X.columns: X[c] = np.nan
            X = X[self.feat_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.medians is None: self.medians = X.median(numeric_only=True).fillna(0.0)
        return X.fillna(self.medians).fillna(0.0)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xn = _numeric_df(X)
        Xn = Xn.loc[:, Xn.notna().any(axis=0)]
        self.feat_cols = Xn.columns.tolist()
        Xp = self._prepare(Xn)
        self.model.fit(Xp, y)
        self.fitted = True
        print(f"[RandomForestModel] fitted | trees={self.params['n_estimators']}, min_leaf={self.params['min_samples_leaf']}, max_features={self.params['max_features']}")
        return self

    def predict(self, X: pd.DataFrame, clip_to_bounds: bool = True) -> np.ndarray:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X)
        yhat = self.model.predict(Xp)
        return _clip(yhat) if clip_to_bounds else yhat

    def predict_interval(self, X: pd.DataFrame, lower: float = 0.1, upper: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X)
        preds = np.stack([t.predict(Xp) for t in self.model.estimators_], axis=1)
        med = np.median(preds, axis=1)
        lo = np.quantile(preds, lower, axis=1)
        hi = np.quantile(preds, upper, axis=1)
        return _clip(med), _clip(lo), _clip(hi)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, label: str = "RandomForest") -> Tuple[float, float]:
        preds = self.predict(X)
        mae, r2 = mean_absolute_error(y, preds), r2_score(y, preds)
        print(f"{label:<12} | MAE: {mae:.3f} | R²: {r2:.3f}")
        return mae, r2

    def feature_importances(self, top_n: int = 15) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        imp = pd.Series(self.model.feature_importances_, index=self.feat_cols)
        return imp.sort_values(ascending=False).head(top_n).round(4)

    def shap_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X)
        bg = _bg_sample(Xp, 500)
        expl = shap.TreeExplainer(self.model, data=bg, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(Xp)
        return _mean_abs_shap(sv, self.feat_cols).head(top_n)

    def sklearn_pipeline(self):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(**self.params))])

    def save(self, path: str):
        joblib.dump({"model": self.model, "medians": self.medians, "feat_cols": self.feat_cols}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.model, self.medians, self.feat_cols = obj["model"], obj["medians"], obj["feat_cols"]
        self.fitted = True
        return self

class XGBoostModel:
    def __init__(self, n_estimators: int = 600, learning_rate: float = 0.05, max_depth: int = 5, subsample: float = 0.9, colsample_bytree: float = 0.9, reg_alpha: float = 0.0, reg_lambda: float = 1.0, random_state: int = 42, n_jobs: int = -1, tree_method: str = "hist"):
        if not _HAS_XGB: raise ImportError("xgboost is not installed. `pip install xgboost`")
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=random_state, n_jobs=n_jobs, tree_method=tree_method, objective="reg:squarederror")
        self.model = _XGBRegressor(**self.params)
        self.medians: Optional[pd.Series] = None
        self.feat_cols: Optional[List[str]] = None
        self.fitted = False

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = _numeric_df(X)
        if self.feat_cols is not None:
            for c in self.feat_cols:
                if c not in Xn.columns: Xn[c] = np.nan
            Xn = Xn[self.feat_cols]
        Xn = Xn.replace([np.inf, -np.inf], np.nan)
        if self.medians is None: self.medians = Xn.median(numeric_only=True).fillna(0.0)
        return Xn.fillna(self.medians).fillna(0.0)

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None, early_stopping_rounds: int = 50):
        Xn = _numeric_df(X)
        self.feat_cols = Xn.columns.tolist()
        Xtr = self._prepare(Xn)
        if X_val is not None and y_val is not None:
            Xv = _numeric_df(X_val)[self.feat_cols]
            Xv = self._prepare(Xv)
            self.model.fit(Xtr, y, eval_set=[(Xtr, y), (Xv, y_val)], verbose=False, early_stopping_rounds=early_stopping_rounds)
        else:
            self.model.fit(Xtr, y, verbose=False)
        self.fitted = True
        best = getattr(self.model, "best_ntree_limit", None)
        print(f"[XGBoostModel] fitted | {'best_trees='+str(best) if best else 'trees='+str(self.params['n_estimators'])}")
        return self

    def predict(self, X: pd.DataFrame, clip_to_bounds: bool = True) -> np.ndarray:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X)
        best = getattr(self.model, "best_ntree_limit", None)
        yhat = self.model.predict(Xp, ntree_limit=best) if best else self.model.predict(Xp)
        return _clip(yhat) if clip_to_bounds else yhat

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, label: str = "XGBoost") -> Tuple[float, float]:
        preds = self.predict(X)
        mae, r2 = mean_absolute_error(y, preds), r2_score(y, preds)
        print(f"{label:<12} | MAE: {mae:.3f} | R²: {r2:.3f}")
        return mae, r2

    def feature_importances(self, importance_type: str = "gain", top_n: int = 20) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        booster = self.model.get_booster()
        imp_values = booster.get_score(importance_type=importance_type)
        mapping = {f"f{i}": c for i, c in enumerate(self.feat_cols)}
        imp = pd.Series({mapping.get(k, k): v for k, v in imp_values.items()})
        return imp.sort_values(ascending=False).head(top_n)

    def shap_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X)
        bg = _bg_sample(Xp, 500)
        expl = shap.TreeExplainer(self.model, data=bg, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(Xp)
        return _mean_abs_shap(sv, self.feat_cols).head(top_n)

    def sklearn_pipeline(self):
        if not _HAS_XGB: raise ImportError("xgboost not available.")
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", _XGBRegressor(**self.params))])

    def save(self, path: str):
        joblib.dump({"model": self.model, "medians": self.medians, "feat_cols": self.feat_cols, "params": self.params}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.model, self.medians, self.feat_cols = obj["model"], obj["medians"], obj["feat_cols"]
        self.params = obj.get("params", self.params)
        self.fitted = True
        return self

class TorchMLPModel:
    def __init__(self,
                 hidden_sizes: List[int] = [128, 64],
                 dropout: float = 0.1,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 64,
                 epochs: int = 50,
                 patience: int = 8,
                 random_state: int = 42,
                 device: Optional[str] = None,
                 use_batchnorm: bool = False,
                 bn_momentum: float = 0.1,
                 scheduler_type: str = "none",
                 scheduler_step: int = 10,
                 scheduler_gamma: float = 0.5):
        if not _HAS_TORCH:
            raise ImportError("PyTorch not available.")
        self._ensure_torch()
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.medians: Optional[pd.Series] = None
        self.feat_cols: Optional[List[str]] = None
        self.fitted = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[Any] = None
        self.use_batchnorm = use_batchnorm
        self.bn_momentum = bn_momentum
        self.scheduler_type = scheduler_type
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.epoch_logs_: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(**cfg)

    def get_params(self) -> dict:
        return {
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "random_state": self.random_state,
            "device": self.device,
            "use_batchnorm": self.use_batchnorm,
            "bn_momentum": self.bn_momentum,
            "scheduler_type": self.scheduler_type,
            "scheduler_step": self.scheduler_step,
            "scheduler_gamma": self.scheduler_gamma,
        }

    def set_params(self, **cfg):
        for k, v in cfg.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    def _ensure_torch(self):
        global torch, nn, TensorDataset, DataLoader
        if torch is None:
            import torch as _t
            torch = _t
            nn = _t.nn
            TensorDataset = _t.utils.data.TensorDataset
            DataLoader = _t.utils.data.DataLoader

    def _prepare(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        X = _numeric_df(X).astype(np.float32)
        if self.feat_cols is not None:
            for c in self.feat_cols:
                if c not in X.columns:
                    X[c] = np.nan
            X = X[self.feat_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.medians is None:
            self.medians = X.median(numeric_only=True).fillna(0.0)
        X = X.fillna(self.medians).fillna(0.0)
        Xs = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        np.clip(Xs, -10, 10, out=Xs)
        return np.nan_to_num(Xs).astype(np.float32)

    def _build_net(self, input_dim: int) -> Any:
        layers, in_dim = [], input_dim
        for h in self.hidden_sizes:
            layers += [nn.Linear(in_dim, h)]
            if self.use_batchnorm:
                layers += [nn.BatchNorm1d(h, momentum=self.bn_momentum)]
            layers += [nn.ReLU()]
            if self.dropout > 0:
                layers += [nn.Dropout(self.dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        return nn.Sequential(*layers)

    def _make_scheduler(self, opt, epochs: int):
        if self.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(opt, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        if self.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        return None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        from tqdm import tqdm
        self._ensure_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        Xn = _numeric_df(X)
        keep = (Xn.notna().any(axis=0)) & (Xn.std(axis=0, skipna=True).fillna(0) > 0)
        Xn = Xn.loc[:, keep]
        self.feat_cols = Xn.columns.tolist()
        Xp = self._prepare(Xn, fit_scaler=True)
        yp = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.net = self._build_net(Xp.shape[1]).to(self.device)
        ds = TensorDataset(torch.from_numpy(Xp), torch.from_numpy(yp))
        bs = min(self.batch_size, len(ds))
        drop_last = self.use_batchnorm
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = self._make_scheduler(opt, self.epochs)
        loss_fn = nn.MSELoss()
        best_loss, best_state, wait = float("inf"), None, 0
        self.epoch_logs_ = []
        for epoch in range(self.epochs):
            self.net.train()
            total, count = 0.0, 0
            for xb, yb in tqdm(loader, desc=f"epoch {epoch+1}/{self.epochs}", leave=False):
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
                count += len(xb)
            epoch_loss = total / max(1, count)
            if sch is not None:
                if self.scheduler_type in ("step", "cosine"):
                    sch.step()
            lr_now = opt.param_groups[0]["lr"]
            self.epoch_logs_.append({
                "epoch": int(epoch + 1),
                "train_loss": float(epoch_loss),
                "lr": float(lr_now),
            })
            if epoch_loss < best_loss - 1e-6:
                best_loss, best_state, wait = epoch_loss, {k: v.cpu() for k, v in self.net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state:
            self.net.load_state_dict(best_state)
        self.fitted = True
        print(f"[TorchMLPModel] fitted | best_train_MSE={best_loss:.6f}")
        return self

    def predict(self, X: pd.DataFrame, clip_to_bounds: bool = True) -> np.ndarray:
        if not self.fitted or self.net is None:
            raise RuntimeError("Model not fitted.")
        self._ensure_torch()
        self.net.eval()
        Xp = self._prepare(X, fit_scaler=False)
        with torch.no_grad():
            preds = self.net(torch.from_numpy(Xp).to(self.device)).cpu().numpy().reshape(-1)
        return _clip(preds) if clip_to_bounds else preds

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, label: str = "TorchMLP") -> Tuple[float, float]:
        preds = self.predict(X)
        mae, r2 = mean_absolute_error(y, preds), r2_score(y, preds)
        print(f"{label:<12} | MAE: {mae:.3f} | R²: {r2:.3f}")
        return mae, r2

    def shap_importance(self, X: pd.DataFrame, top_n: int = 20, bg_size: int = 200) -> pd.Series:
        if not self.fitted or self.net is None:
            raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X, fit_scaler=False)
        bg = _bg_sample(Xp, bg_size)
        def f(x):
            with torch.no_grad():
                xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
                yt = self.net(xt.to(self.device)).cpu().numpy().reshape(-1)
            return yt
        expl = shap.KernelExplainer(f, bg)
        sv = expl.shap_values(Xp, nsamples="auto")
        return _mean_abs_shap(sv, self.feat_cols).head(top_n)

    def save(self, path: str):
        if not self.fitted or self.net is None:
            raise RuntimeError("Model not fitted.")
        joblib.dump({
            "scaler": self.scaler,
            "medians": self.medians,
            "feat_cols": self.feat_cols,
            "state_dict": {k: v.cpu().numpy() for k, v in self.net.state_dict().items()},
            "hparams": self.get_params(),
            "epoch_logs_": self.epoch_logs_,
        }, path)

    def load(self, path: str):
        if not _HAS_TORCH:
            raise ImportError("PyTorch not available.")
        self._ensure_torch()
        obj = joblib.load(path)
        self.scaler, self.medians, self.feat_cols = obj["scaler"], obj["medians"], obj["feat_cols"]
        self.set_params(**obj.get("hparams", {}))
        self.net = self._build_net(len(self.feat_cols)).to(self.device)
        state = {k: torch.from_numpy(v) for k, v in obj["state_dict"].items()}
        self.net.load_state_dict(state)
        self.epoch_logs_ = obj.get("epoch_logs_", [])
        self.fitted = True
        return self

class SVRModel:
    def __init__(self, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1, gamma: str = "scale", degree: int = 3, coef0: float = 0.0):
        self.params = dict(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, degree=degree, coef0=coef0)
        self.model = SVR(**self.params)
        self.scaler = StandardScaler()
        self.medians: Optional[pd.Series] = None
        self.feat_cols: Optional[List[str]] = None
        self.fitted = False

    def _prepare(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        X = _numeric_df(X).astype(np.float64)
        if self.feat_cols is not None:
            for c in self.feat_cols:
                if c not in X.columns: X[c] = np.nan
            X = X[self.feat_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.medians is None: self.medians = X.median(numeric_only=True).fillna(0.0)
        X = X.fillna(self.medians).fillna(0.0)
        Xs = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        np.clip(Xs, -10, 10, out=Xs)
        return np.nan_to_num(Xs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xn = _numeric_df(X)
        keep = (Xn.notna().any(axis=0)) & (Xn.std(axis=0, skipna=True).fillna(0) > 0)
        Xn = Xn.loc[:, keep]
        self.feat_cols = Xn.columns.tolist()
        Xp = self._prepare(Xn, fit_scaler=True)
        self.model.fit(Xp, y)
        self.fitted = True
        print(f"[SVRModel] fitted | kernel={self.params['kernel']} C={self.params['C']} eps={self.params['epsilon']}")
        return self

    def predict(self, X: pd.DataFrame, clip_to_bounds: bool = True) -> np.ndarray:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X, fit_scaler=False)
        preds = self.model.predict(Xp)
        return _clip(preds) if clip_to_bounds else preds

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, label: str = "SVR") -> Tuple[float, float]:
        preds = self.predict(X)
        mae, r2 = mean_absolute_error(y, preds), r2_score(y, preds)
        print(f"{label:<12} | MAE: {mae:.3f} | R²: {r2:.3f}")
        return mae, r2

    def shap_importance(self, X: pd.DataFrame, top_n: int = 20, bg_size: int = 200) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X, fit_scaler=False)
        bg = _bg_sample(Xp, bg_size)
        expl = shap.KernelExplainer(self.model.predict, bg)
        sv = expl.shap_values(Xp, nsamples="auto")
        return _mean_abs_shap(sv, self.feat_cols).head(top_n)

    def sklearn_pipeline(self):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVR(**self.params))])

    def save(self, path: str):
        joblib.dump({"scaler": self.scaler, "model": self.model, "medians": self.medians, "feat_cols": self.feat_cols, "params": self.params}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.scaler, self.model, self.medians, self.feat_cols = obj["scaler"], obj["model"], obj["medians"], obj["feat_cols"]
        self.params = obj.get("params", self.params)
        self.fitted = True
        return self

class KNNRegressorModel:
    def __init__(self, n_neighbors: int = 5, weights: str = "distance", metric: str = "minkowski", p: int = 2, n_jobs: int = -1):
        self.params = dict(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p, n_jobs=n_jobs)
        self.model = KNeighborsRegressor(**self.params)
        self.scaler = StandardScaler()
        self.medians: Optional[pd.Series] = None
        self.feat_cols: Optional[List[str]] = None
        self.fitted = False

    def _prepare(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        X = _numeric_df(X).astype(np.float64)
        if self.feat_cols is not None:
            for c in self.feat_cols:
                if c not in X.columns: X[c] = np.nan
            X = X[self.feat_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.medians is None: self.medians = X.median(numeric_only=True).fillna(0.0)
        X = X.fillna(self.medians).fillna(0.0)
        Xs = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        np.clip(Xs, -10, 10, out=Xs)
        return np.nan_to_num(Xs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xn = _numeric_df(X)
        keep = (Xn.notna().any(axis=0)) & (Xn.std(axis=0, skipna=True).fillna(0) > 0)
        Xn = Xn.loc[:, keep]
        self.feat_cols = Xn.columns.tolist()
        Xp = self._prepare(Xn, fit_scaler=True)
        self.model.fit(Xp, y)
        self.fitted = True
        print(f"[KNNRegressorModel] fitted | k={self.params['n_neighbors']} weights={self.params['weights']}")
        return self

    def predict(self, X: pd.DataFrame, clip_to_bounds: bool = True) -> np.ndarray:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X, fit_scaler=False)
        preds = self.model.predict(Xp)
        return _clip(preds) if clip_to_bounds else preds

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, label: str = "KNNRegressor") -> Tuple[float, float]:
        preds = self.predict(X)
        mae, r2 = mean_absolute_error(y, preds), r2_score(y, preds)
        print(f"{label:<12} | MAE: {mae:.3f} | R²: {r2:.3f}")
        return mae, r2

    def shap_importance(self, X: pd.DataFrame, top_n: int = 20, bg_size: int = 200) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X, fit_scaler=False)
        bg = _bg_sample(Xp, bg_size)
        expl = shap.KernelExplainer(self.model.predict, bg)
        sv = expl.shap_values(Xp, nsamples="auto")
        return _mean_abs_shap(sv, self.feat_cols).head(top_n)

    def sklearn_pipeline(self):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", KNeighborsRegressor(**self.params))])

    def save(self, path: str):
        joblib.dump({"scaler": self.scaler, "model": self.model, "medians": self.medians, "feat_cols": self.feat_cols, "params": self.params}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.scaler, self.model, self.medians, self.feat_cols = obj["scaler"], obj["model"], obj["medians"], obj["feat_cols"]
        self.params = obj.get("params", self.params)
        self.fitted = True
        return self

class CatBoostRegressorModel:
    def __init__(self, iterations: int = 600, learning_rate: float = 0.05, depth: int = 6, l2_leaf_reg: float = 3.0, subsample: float = 0.9, random_state: int = 42, early_stopping_rounds: int = 50, use_gpu: bool = False, verbose: bool = False):
        if not _HAS_CATBOOST: raise ImportError("catboost is not installed. `pip install catboost`")
        self.params = dict(iterations=iterations, learning_rate=learning_rate, depth=depth, l2_leaf_reg=l2_leaf_reg, subsample=subsample, random_seed=random_state, loss_function="RMSE", task_type="GPU" if use_gpu else "CPU", early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        self.model = CatBoostRegressor(**self.params)
        self.scaler = StandardScaler()
        self.medians: Optional[pd.Series] = None
        self.feat_cols: Optional[List[str]] = None
        self.fitted = False

    def _prepare(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        X = _numeric_df(X).astype(np.float64)
        if self.feat_cols is not None:
            for c in self.feat_cols:
                if c not in X.columns: X[c] = np.nan
            X = X[self.feat_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.medians is None: self.medians = X.median(numeric_only=True).fillna(0.0)
        X = X.fillna(self.medians).fillna(0.0)
        Xs = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        np.clip(Xs, -10, 10, out=Xs)
        return np.nan_to_num(Xs)

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None):
        Xn = _numeric_df(X)
        keep = (Xn.notna().any(axis=0)) & (Xn.std(axis=0, skipna=True).fillna(0) > 0)
        Xn = Xn.loc[:, keep]
        self.feat_cols = Xn.columns.tolist()
        Xp = self._prepare(Xn, fit_scaler=True)
        if X_val is not None and y_val is not None:
            Xv = _numeric_df(X_val)[self.feat_cols]
            Xv = self._prepare(Xv, fit_scaler=False)
            self.model.fit(Xp, y, eval_set=(Xv, y_val))
        else:
            self.model.fit(Xp, y)
        self.fitted = True
        print(f"[CatBoostRegressorModel] fitted | depth={self.params['depth']} lr={self.params['learning_rate']}")
        return self

    def predict(self, X: pd.DataFrame, clip_to_bounds: bool = True) -> np.ndarray:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X, fit_scaler=False)
        preds = self.model.predict(Xp)
        return _clip(preds) if clip_to_bounds else preds

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, label: str = "CatBoost") -> Tuple[float, float]:
        preds = self.predict(X)
        mae, r2 = mean_absolute_error(y, preds), r2_score(y, preds)
        print(f"{label:<12} | MAE: {mae:.3f} | R²: {r2:.3f}")
        return mae, r2

    def feature_importances(self, top_n: int = 15) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        imp = pd.Series(self.model.get_feature_importance(), index=self.feat_cols)
        return imp.sort_values(ascending=False).head(top_n).round(4)

    def shap_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.Series:
        if not self.fitted: raise RuntimeError("Model not fitted.")
        Xp = self._prepare(X, fit_scaler=False)
        bg = _bg_sample(Xp, 500)
        expl = shap.TreeExplainer(self.model, data=bg, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(Xp)
        return _mean_abs_shap(sv, self.feat_cols).head(top_n)

    def sklearn_pipeline(self):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", CatBoostRegressor(**self.params))])

    def save(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler, "medians": self.medians, "feat_cols": self.feat_cols, "params": self.params}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.model, self.scaler, self.medians, self.feat_cols = obj["model"], obj["scaler"], obj["medians"], obj["feat_cols"]
        self.params = obj.get("params", self.params)
        self.fitted = True
        return self
