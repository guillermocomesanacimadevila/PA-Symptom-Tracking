import os
import joblib
import numpy as np
import pandas as pd
import warnings
from utils import (
    load_ml_ready,
    ensure_horizon_label,
    split_by_id_time,
    build_feature_label_matrices,
    evaluate_regression,
    run_grid_search,
    shap_importance_for_pipeline,
    predict_pipeline_compat,
    BASE_COMPOSITE_COL,
    make_pipeline_and_grid,
    gpu_available,
    cuml_available,
)

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = "../Data/Symptomtrackingdata_csv-cleaned_with_vars_ml_ready.csv"
OUT_DIR = "../grid_results_horizons"
ID_COL = "id"
DAY_COL = "day"
MODEL_LIST = ["catboost", "mlp"]
# MODEL_LIST = ["lasso","elasticnet","rf","svr","knn","xgb","catboost","mlp"]
H_MAX = 14
USE_DELTA = True
N_CALIB_BINS = 10
SHAP_HORIZONS = [1, 3, 7, 10, 14]

def _gpu_summary():
    try:
        import torch
        use_cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
        print(f"[DEVICE] GPU available={use_cuda} | device={name}")
    except Exception:
        print("[DEVICE] GPU available=False | device=CPU")
    try:
        import cuml
        print("[RAPIDS] cuML available=True")
    except Exception:
        print("[RAPIDS] cuML available=False")
    try:
        import xgboost as xgb
        print(f"[XGBoost] version={xgb.__version__}")
    except Exception:
        print("[XGBoost] not available")
    try:
        import catboost
        print(f"[CatBoost] version={catboost.__version__}")
    except Exception:
        print("[CatBoost] not available")

def _use_gpu_for_model(name: str) -> bool:
    n = name.lower()
    if n in ("rf", "svr", "knn", "lasso", "elasticnet"):
        return gpu_available() and cuml_available()
    if n in ("xgb", "catboost"):
        return gpu_available()
    return False

def _calibration_bins(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if len(df) == 0:
        return pd.DataFrame(columns=["bin","count","y_pred_mean","y_true_mean","y_pred_std","y_true_std","y_pred_min","y_pred_max"])
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(df["y_pred"].to_numpy(), q))
    if len(edges) < 3:
        edges = np.linspace(df["y_pred"].min(), df["y_pred"].max(), min(n_bins, max(2, len(df)//5)) + 1)
    df["bin"] = np.digitize(df["y_pred"], edges[1:-1], right=False)
    g = df.groupby("bin", as_index=False)
    out = g.agg(
        count=("y_pred","size"),
        y_pred_mean=("y_pred","mean"),
        y_true_mean=("y_true","mean"),
        y_pred_std=("y_pred","std"),
        y_true_std=("y_true","std"),
        y_pred_min=("y_pred","min"),
        y_pred_max=("y_pred","max"),
    )
    out["bin"] = out["bin"].astype(int)
    return out.sort_values("bin").reset_index(drop=True)

def _calibration_summary(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[m]
    yp = y_pred[m]
    if len(yt) == 0:
        return pd.DataFrame([{"n":0,"intercept":np.nan,"slope":np.nan,"MAE":np.nan,"RMSE":np.nan,"corr":np.nan}])
    A = np.vstack([yp, np.ones_like(yp)]).T
    slope, intercept = np.linalg.lstsq(A, yt, rcond=None)[0]
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp)**2))
    corr = np.corrcoef(yp, yt)[0,1] if len(yt) > 1 else np.nan
    return pd.DataFrame([{"n":int(len(yt)),"intercept":float(intercept),"slope":float(slope),"MAE":float(mae),"RMSE":float(rmse),"corr":float(corr)}])

def main():
    _gpu_summary()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_ml_ready(DATA_PATH)
    all_rows = []
    for name in MODEL_LIST:
        print("=" * 60)
        print(f"MODEL {name}")
        print("=" * 60)
        best_params = None
        best_cv_csv = None
        use_gpu = _use_gpu_for_model(name)
        for h in range(1, H_MAX + 1):
            print("-" * 40)
            print(f"model={name} h={h}")
            print("-" * 40)
            df_h, label_col = ensure_horizon_label(df, h=h, id_col=ID_COL, day_col=DAY_COL, base_col=BASE_COMPOSITE_COL, use_delta=USE_DELTA)
            train, val, test = split_by_id_time(df_h, id_col=ID_COL, day_col=DAY_COL, test_size=0.15, val_size=0.15)
            Xtr, ytr, feat_cols = build_feature_label_matrices(train, id_col=ID_COL, day_col=DAY_COL, label_col=label_col)
            Xva, yva, _ = build_feature_label_matrices(val, id_col=ID_COL, day_col=DAY_COL, label_col=label_col)
            Xt, yt, _ = build_feature_label_matrices(test, id_col=ID_COL, day_col=DAY_COL, label_col=label_col)
            Xcv = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True)
            ycv = np.concatenate([ytr, yva])
            if BASE_COMPOSITE_COL not in test.columns:
                print(f"skip model={name} h={h} missing U_t")
                continue
            Ut_test = test[BASE_COMPOSITE_COL].to_numpy()
            abs_col = f"Y_next_h{h}"
            if abs_col not in test.columns:
                print(f"skip model={name} h={h} missing {abs_col}")
                continue
            ytrue_abs = test[abs_col].to_numpy()
            out_dir_h = os.path.join(OUT_DIR, name, f"h{h}")
            os.makedirs(out_dir_h, exist_ok=True)
            if h == 1:
                try:
                    summary = run_grid_search(model_name=name, X=Xcv, y=ycv, cv_splits=5, out_dir=out_dir_h, refit_metric="r2")
                except ImportError as e:
                    print(f"{name} h=1 skipped: {e}")
                    break
                best_params = summary.get("best_params", None)
                best_cv_csv = summary.get("cv_results_csv", None)
                best_path = summary["best_model_path"]
                if name == "mlp":
                    from models import TorchMLPModel
                    best_model = TorchMLPModel().load(best_path)
                else:
                    best_model = joblib.load(best_path)
                best_model_path = best_path
            else:
                if best_params is None:
                    print(f"{name} h={h} skipped no best_params from h=1")
                    continue
                if name == "mlp":
                    from models import TorchMLPModel
                    best_model = TorchMLPModel.from_config(best_params)
                    best_model.fit(Xcv, ycv)
                    best_model_path = os.path.join(out_dir_h, f"{name}_best.pkl")
                    best_model.save(best_model_path)
                else:
                    est, _ = make_pipeline_and_grid(name, use_gpu)
                    est.set_params(**best_params)
                    est.fit(Xcv, ycv)
                    best_model = est
                    best_model_path = os.path.join(out_dir_h, f"{name}_best.pkl")
                    joblib.dump(best_model, best_model_path)
            yhat_target = predict_pipeline_compat(best_model, Xt)
            if USE_DELTA:
                yhat_abs = np.clip(Ut_test + yhat_target, 0.0, 10.0)
            else:
                yhat_abs = np.clip(yhat_target, 0.0, 10.0)
            target_metrics = evaluate_regression(yt, yhat_target)
            abs_metrics = evaluate_regression(ytrue_abs, yhat_abs)
            naive_pred = np.clip(Ut_test, 0.0, 10.0)
            naive_metrics = evaluate_regression(ytrue_abs, naive_pred)
            print(
                f"TARGET_R2={target_metrics['R2']:.3f} "
                f"TARGET_MAE={target_metrics['MAE']:.3f} "
                f"Y_R2={abs_metrics['R2']:.3f} "
                f"Y_MAE={abs_metrics['MAE']:.3f} "
                f"Naive_Y_R2={naive_metrics['R2']:.3f} "
                f"Naive_Y_MAE={naive_metrics['MAE']:.3f}"
            )
            row = {
                "model": name,
                "h": h,
                "label": label_col,
                "TARGET_MAE": target_metrics["MAE"],
                "TARGET_RMSE": target_metrics["RMSE"],
                "TARGET_R2": target_metrics["R2"],
                "Y_MAE": abs_metrics["MAE"],
                "Y_RMSE": abs_metrics["RMSE"],
                "Y_R2": abs_metrics["R2"],
                "Naive_Y_MAE": naive_metrics["MAE"],
                "Naive_Y_RMSE": naive_metrics["RMSE"],
                "Naive_Y_R2": naive_metrics["R2"],
                "best_model_path": best_model_path,
                "cv_results_csv": best_cv_csv,
            }
            all_rows.append(row)
            preds_path = os.path.join(out_dir_h, "predictions.csv")
            pred_df = pd.DataFrame({
                ID_COL: test[ID_COL].to_numpy(),
                DAY_COL: test[DAY_COL].to_numpy(),
                "y_true_target": yt,
                "y_pred_target": yhat_target,
                "y_true_abs": ytrue_abs,
                "y_pred_abs": yhat_abs,
                "y_naive_abs": naive_pred,
            })
            pred_df.to_csv(preds_path, index=False)
            bins_target = _calibration_bins(yt, yhat_target, n_bins=N_CALIB_BINS)
            bins_abs = _calibration_bins(ytrue_abs, yhat_abs, n_bins=N_CALIB_BINS)
            bins_target.to_csv(os.path.join(out_dir_h, "calibration_bins_target.csv"), index=False)
            bins_abs.to_csv(os.path.join(out_dir_h, "calibration_bins_abs.csv"), index=False)
            summ_target = _calibration_summary(yt, yhat_target)
            summ_abs = _calibration_summary(ytrue_abs, yhat_abs)
            summ_target.to_csv(os.path.join(out_dir_h, "calibration_summary_target.csv"), index=False)
            summ_abs.to_csv(os.path.join(out_dir_h, "calibration_summary_abs.csv"), index=False)
            if h in SHAP_HORIZONS:
                try:
                    X_for_shap = Xcv.sample(n=min(50, len(Xcv)), random_state=42).copy()
                    if name == "mlp":
                        shap_imp = best_model.shap_importance(X_for_shap, top_n=30)
                    else:
                        shap_imp = shap_importance_for_pipeline(best_model, X_for_shap, top_n=30)
                    imp_path = os.path.join(OUT_DIR, name, f"h{h}_shap_importance.csv")
                    shap_imp.to_csv(imp_path, header=["mean_|SHAP|"])
                    print(f"SHAP -> {imp_path}")
                except Exception as e:
                    print(f"SHAP skipped (error): {e}")
            else:
                print(f"SHAP skipped for model={name} h={h} (not in SHAP_HORIZONS)")
    summary_path = os.path.join(OUT_DIR, "summary_all_models_all_horizons.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"summary -> {summary_path}")

if __name__ == "__main__":
    main()
