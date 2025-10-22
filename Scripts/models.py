# Lasso | ElasticNet | RandomForest | XGBoost | Torch MLP | SVR | KNN | CatBoost

import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple, List
from sklearn.linear_model import LassoCV, Lasso, ElasticNetCV, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor as _XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    _XGBRegressor = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    torch = None
    nn = None
    TensorDataset = None
    DataLoader = None

try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False
    CatBoostRegressor = None


# ========================================================== #
# ------------------ GLOBAL SMALL UTILITIES ---------------- #
# ========================================================== #

TARGET_MIN, TARGET_MAX = 0.0, 10.0

def _clip(yhat: np.ndarray, lo: float = TARGET_MIN, hi: float = TARGET_MAX) -> np.ndarray:
    return np.clip(yhat, lo, hi)

def _numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    return X.select_dtypes(include=[np.number]).copy()


# ========================================================== #
# ---------------------- LASSO MODEL ----------------------- #
# ========================================================== #

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


# ========================================================== #
# -------------------- ELASTIC NET MODEL ------------------- #
# ========================================================== #

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


# ========================================================== #
# ----------------- RANDOM FOREST MODEL -------------------- #
# ========================================================== #

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

    def sklearn_pipeline(self):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(**self.params))])

    def save(self, path: str):
        joblib.dump({"model": self.model, "medians": self.medians, "feat_cols": self.feat_cols}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.model, self.medians, self.feat_cols = obj["model"], obj["medians"], obj["feat_cols"]
        self.fitted = True
        return self


# ========================================================== #
# --------------------- XGBOOST MODEL ---------------------- #
# ========================================================== #

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


# ========================================================== #
# ------------------- PYTORCH MLP MODEL -------------------- #
# ========================================================== #

class TorchMLPModel:
    def __init__(self, hidden_sizes: List[int] = [64, 32], dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 64, epochs: int = 3, patience: int = 10, random_state: int = 42, device: Optional[str] = None):
        if not _HAS_TORCH: raise ImportError("PyTorch not available. Install torch.")
        self.hidden_sizes, self.dropout = hidden_sizes, dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.epochs, self.patience = batch_size, epochs, patience
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.medians: Optional[pd.Series] = None
        self.feat_cols: Optional[List[str]] = None
        self.fitted = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[nn.Module] = None

    def _prepare(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        X = _numeric_df(X).astype(np.float32)
        if self.feat_cols is not None:
            for c in self.feat_cols:
                if c not in X.columns: X[c] = np.nan
            X = X[self.feat_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.medians is None: self.medians = X.median(numeric_only=True).fillna(0.0)
        X = X.fillna(self.medians).fillna(0.0)
        Xs = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        np.clip(Xs, -10, 10, out=Xs)
        return np.nan_to_num(Xs).astype(np.float32)

    def _build_net(self, input_dim: int) -> nn.Module:
        layers, in_dim = [], input_dim
        for h in self.hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if self.dropout > 0: layers += [nn.Dropout(self.dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        return nn.Sequential(*layers)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        torch.manual_seed(self.random_state); np.random.seed(self.random_state)
        Xn = _numeric_df(X)
        keep = (Xn.notna().any(axis=0)) & (Xn.std(axis=0, skipna=True).fillna(0) > 0)
        Xn = Xn.loc[:, keep]
        self.feat_cols = Xn.columns.tolist()
        Xp = self._prepare(Xn, fit_scaler=True)
        yp = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.net = self._build_net(Xp.shape[1]).to(self.device)
        ds = TensorDataset(torch.from_numpy(Xp), torch.from_numpy(yp))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_loss, best_state, wait = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.net.train()
            total, count = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward(); opt.step()
                total += loss.item() * len(xb); count += len(xb)
            epoch_loss = total / max(1, count)
            if epoch_loss < best_loss - 1e-6:
                best_loss, best_state, wait = epoch_loss, {k: v.cpu() for k, v in self.net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience: break

        if best_state: self.net.load_state_dict(best_state)
        self.fitted = True
        print(f"[TorchMLPModel] fitted | best_train_MSE={best_loss:.6f}")
        return self

    def predict(self, X: pd.DataFrame, clip_to_bounds: bool = True) -> np.ndarray:
        if not self.fitted or self.net is None: raise RuntimeError("Model not fitted.")
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

    def save(self, path: str):
        if not self.fitted or self.net is None: raise RuntimeError("Model not fitted.")
        joblib.dump({
            "scaler": self.scaler,
            "medians": self.medians,
            "feat_cols": self.feat_cols,
            "state_dict": {k: v.cpu().numpy() for k, v in self.net.state_dict().items()},
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout
        }, path)

    def load(self, path: str):
        if not _HAS_TORCH: raise ImportError("PyTorch not available.")
        obj = joblib.load(path)
        self.scaler, self.medians, self.feat_cols = obj["scaler"], obj["medians"], obj["feat_cols"]
        self.hidden_sizes = obj.get("hidden_sizes", self.hidden_sizes)
        self.dropout = obj.get("dropout", self.dropout)
        self.net = self._build_net(len(self.feat_cols)).to(self.device)
        state = {k: torch.from_numpy(v) for k, v in obj["state_dict"].items()}
        self.net.load_state_dict(state)
        self.fitted = True
        return self


# ========================================================== #
# ---------------- SUPPORT VECTOR REGRESSOR ---------------- #
# ========================================================== #

class SVRModel:
    """SVR with scaling + median imputation."""
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


# ========================================================== #
# ---------------- K-NEIGHBOURS REGRESSOR ------------------ #
# ========================================================== #

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


# ========================================================== #
# ------------------ CATBOOST REGRESSOR -------------------- #
# ========================================================== #

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

    def sklearn_pipeline(self):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", CatBoostRegressor(**self.params))])

    def save(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler, "medians": self.medians, "feat_cols": self.feat_cols, "params": self.params}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.model, self.scaler, self.medians, self.feat_cols = obj["model"], obj["scaler"], obj["medians"], obj["feat_cols"]
        self.params = obj.get("params", self.params)
        self.fitted = True
        return self
