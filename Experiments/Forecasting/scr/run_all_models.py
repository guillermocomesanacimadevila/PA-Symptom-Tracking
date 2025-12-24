import os
import argparse
import multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
import warnings
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from utils import (
    load_ml_ready,
    ensure_horizon_label,
    split_by_id_time,
    build_feature_label_matrices,
    evaluate_regression,
    run_grid_search,
    shap_importance_for_pipeline,
    predict_pipeline_compat,
    make_pipeline_and_grid,
    gpu_available,
    cuml_available,
)

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = "../Data/Symptomtrackingdata_csv-cleaned_with_vars_ml_ready.csv"
OUT_DIR = "../grid_results_horizons"
ID_COL = "id"
DAY_COL = "day"
MODEL_LIST = ["ridge"]
USE_DELTA = True
N_CALIB_BINS = 10
N_BOOT = 200
BOOT_SEED = 42
CI_ALPHA = 0.05
H_MAX_CAP = 60
MIN_ROWS_PER_H = 200


def _use_gpu_for_model(name):
    n = name.lower()
    if n in ("rf", "svr", "knn", "lasso", "elasticnet"):
        return gpu_available() and cuml_available()
    if n in ("xgb", "catboost"):
        return gpu_available()
    return False


def _calibration_bins(y_true, y_pred, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "bin",
                "count",
                "y_pred_mean",
                "y_true_mean",
                "y_pred_std",
                "y_true_std",
                "y_pred_min",
                "y_pred_max",
            ]
        )
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(df["y_pred"].to_numpy(), q))
    if len(edges) < 3:
        edges = np.linspace(
            df["y_pred"].min(),
            df["y_pred"].max(),
            min(n_bins, max(2, len(df) // 5)) + 1,
        )
    df["bin"] = np.digitize(df["y_pred"], edges[1:-1], right=False)
    g = df.groupby("bin", as_index=False)
    out = g.agg(
        count=("y_pred", "size"),
        y_pred_mean=("y_pred", "mean"),
        y_true_mean=("y_true", "mean"),
        y_pred_std=("y_pred", "std"),
        y_true_std=("y_true", "std"),
        y_pred_min=("y_pred", "min"),
        y_pred_max=("y_pred", "max"),
    )
    out["bin"] = out["bin"].astype(int)
    return out.sort_values("bin").reset_index(drop=True)


def _calibration_summary(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[m]
    yp = y_pred[m]
    if len(yt) == 0:
        return pd.DataFrame(
            [
                {
                    "n": 0,
                    "intercept": np.nan,
                    "slope": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "corr": np.nan,
                }
            ]
        )
    A = np.vstack([yp, np.ones_like(yp)]).T
    slope, intercept = np.linalg.lstsq(A, yt, rcond=None)[0]
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    corr = np.corrcoef(yp, yt)[0, 1] if len(yt) > 1 else np.nan
    return pd.DataFrame(
        [
            {
                "n": int(len(yt)),
                "intercept": float(intercept),
                "slope": float(slope),
                "MAE": float(mae),
                "RMSE": float(rmse),
                "corr": float(corr),
            }
        ]
    )


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if len(y_true) < 2:
        return np.nan
    yt = y_true - np.mean(y_true)
    ss_tot = float(np.sum(yt * yt))
    if ss_tot <= 0:
        return np.nan
    resid = y_true - y_pred
    ss_res = float(np.sum(resid * resid))
    return 1.0 - (ss_res / ss_tot)


def _mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def bootstrap_metric_ci_by_id(ids, y_true, y_pred, n_boot=200, seed=42, alpha=0.05):
    ids = np.asarray(ids)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    uniq = pd.unique(ids)
    uniq = uniq[~pd.isna(uniq)]
    if len(uniq) == 0:
        return {
            "Y_R2_ci_low": np.nan,
            "Y_R2_ci_high": np.nan,
            "Y_MAE_ci_low": np.nan,
            "Y_MAE_ci_high": np.nan,
            "n_boot": 0,
        }
    rng = np.random.default_rng(seed)
    r2_vals = np.zeros(int(n_boot), dtype=float)
    mae_vals = np.zeros(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(ids, samp)
        r2_vals[b] = _r2(y_true[mask], y_pred[mask])
        mae_vals[b] = _mae(y_true[mask], y_pred[mask])
    r2_vals = r2_vals[np.isfinite(r2_vals)]
    mae_vals = mae_vals[np.isfinite(mae_vals)]
    a = float(alpha) / 2.0
    out = {"n_boot": int(n_boot)}
    out["Y_R2_ci_low"] = float(np.quantile(r2_vals, a)) if len(r2_vals) else np.nan
    out["Y_R2_ci_high"] = float(np.quantile(r2_vals, 1.0 - a)) if len(r2_vals) else np.nan
    out["Y_MAE_ci_low"] = float(np.quantile(mae_vals, a)) if len(mae_vals) else np.nan
    out["Y_MAE_ci_high"] = float(np.quantile(mae_vals, 1.0 - a)) if len(mae_vals) else np.nan
    return out


def _fit_best_model(name, best_params, use_gpu, X, y):
    if name == "mlp":
        from models import TorchMLPModel

        m = TorchMLPModel.from_config(best_params)
        m.fit(X, y)
        return m
    est, _ = make_pipeline_and_grid(name, use_gpu)
    est.set_params(**best_params)
    est.fit(X, y)
    return est


def _bootstrap_pred_intervals(name, best_params, use_gpu, Xcv, ycv, ids_cv, Xt, base_t_test, use_delta):
    rng = np.random.default_rng(BOOT_SEED)
    ids_cv = np.asarray(ids_cv)
    uniq = pd.unique(ids_cv)
    uniq = uniq[~pd.isna(uniq)]
    idx_map = {}
    for i, pid in enumerate(ids_cv):
        if pid in idx_map:
            idx_map[pid].append(i)
        else:
            idx_map[pid] = [i]
    P_abs = np.zeros((int(N_BOOT), len(Xt)), dtype=float)
    for b in range(int(N_BOOT)):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idxs = []
        for pid in samp:
            idxs.extend(idx_map[pid])
        Xb = Xcv.iloc[idxs]
        yb = ycv[idxs]
        mb = _fit_best_model(name, best_params, use_gpu, Xb, yb)
        yb_target = predict_pipeline_compat(mb, Xt)
        if use_delta:
            yb_abs = np.clip(base_t_test + yb_target, 0.0, 10.0)
        else:
            yb_abs = np.clip(yb_target, 0.0, 10.0)
        P_abs[b] = yb_abs
    a = float(CI_ALPHA) / 2.0
    abs_lo = np.quantile(P_abs, a, axis=0)
    abs_hi = np.quantile(P_abs, 1.0 - a, axis=0)
    return abs_lo, abs_hi


def summarize_predictions(pred_df, id_col, day_col, by):
    if by == "patient":
        key = id_col
        key_name = "patient"
    else:
        key = day_col
        key_name = "day"
    rows = []
    for k, g in pred_df.groupby(key):
        yt = g["y_true_abs"].to_numpy()
        yp = g["y_pred_abs"].to_numpy()
        rows.append(
            {
                key_name: k,
                "n": int(len(g)),
                "y_true_mean": float(np.nanmean(yt)) if len(g) else np.nan,
                "y_pred_mean": float(np.nanmean(yp)) if len(g) else np.nan,
                "mae": _mae(yt, yp),
                "r2": _r2(yt, yp),
                "y_pred_ci_low_mean": float(np.nanmean(g["y_pred_abs_ci_low"].to_numpy()))
                if "y_pred_abs_ci_low" in g.columns
                else np.nan,
                "y_pred_ci_high_mean": float(np.nanmean(g["y_pred_abs_ci_high"].to_numpy()))
                if "y_pred_abs_ci_high" in g.columns
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("n", ascending=False).reset_index(drop=True)
    return out


def _target_config(t):
    t = str(t).strip().upper()
    if t == "U":
        return {
            "name": "U",
            "base_col": "U_t",
            "abs_prefix": "Y_next_h",
            "delta_prefix": "dY_next_h",
            "abs_cap": (0.0, 10.0),
        }
    if t == "W":
        return {
            "name": "W",
            "base_col": "W_t",
            "abs_prefix": "W_next_h",
            "delta_prefix": "dW_next_h",
            "abs_cap": (0.0, 10.0),
        }
    if t == "S":
        return {
            "name": "S",
            "base_col": "S_t",
            "abs_prefix": "S_next_h",
            "delta_prefix": "dS_next_h",
            "abs_cap": (0.0, 10.0),
        }
    raise ValueError("target must be one of: U, W, S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--target", default="U", choices=["U", "W", "S"])
    ap.add_argument("--summarise-by", default="none", choices=["none", "patient", "day", "both"])
    ap.add_argument("--h-max", default="auto")
    ap.add_argument("--h-cap", type=int, default=H_MAX_CAP)
    ap.add_argument("--min-rows-per-h", type=int, default=MIN_ROWS_PER_H)
    args = ap.parse_args()

    cfg = _target_config(args.target)
    base_col = cfg["base_col"]
    abs_prefix = cfg["abs_prefix"]
    delta_prefix = cfg["delta_prefix"]
    lo_cap, hi_cap = cfg["abs_cap"]

    data_path = args.data
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    df = load_ml_ready(data_path)

    h_cols = [c for c in df.columns if c.startswith(abs_prefix)]
    h_found = max([int(c.split(abs_prefix)[1]) for c in h_cols], default=1)
    H_MAX = min(int(args.h_cap), int(h_found))
    if str(args.h_max).lower() != "auto":
        H_MAX = min(H_MAX, int(args.h_max))

    all_rows = []
    for name in MODEL_LIST:
        best_params = None
        best_cv_csv = None
        use_gpu = _use_gpu_for_model(name)
        for h in range(1, H_MAX + 1):
            label_col = f"{delta_prefix}{h}" if USE_DELTA else f"{abs_prefix}{h}"
            if label_col not in df.columns:
                continue

            need_cols = [ID_COL, DAY_COL, label_col, base_col, f"{abs_prefix}{h}"]
            missing = [c for c in need_cols if c not in df.columns]
            if missing:
                continue

            df_h = df.loc[:, need_cols + [c for c in df.columns if c not in need_cols]].copy()
            df_h = df_h.replace([np.inf, -np.inf], np.nan).dropna(subset=[ID_COL, DAY_COL, label_col, base_col, f"{abs_prefix}{h}"])

            if len(df_h) < int(args.min_rows_per_h):
                continue

            train, val, test = split_by_id_time(
                df_h, id_col=ID_COL, day_col=DAY_COL, test_size=0.15, val_size=0.15
            )

            Xtr, ytr, _ = build_feature_label_matrices(train, id_col=ID_COL, day_col=DAY_COL, label_col=label_col)
            Xva, yva, _ = build_feature_label_matrices(val, id_col=ID_COL, day_col=DAY_COL, label_col=label_col)
            Xt, yt, _ = build_feature_label_matrices(test, id_col=ID_COL, day_col=DAY_COL, label_col=label_col)

            Xcv = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True)
            ycv = np.concatenate([ytr, yva])
            ids_cv = np.concatenate([train[ID_COL].to_numpy(), val[ID_COL].to_numpy()])

            base_t_test = test[base_col].to_numpy(dtype=float)
            abs_col = f"{abs_prefix}{h}"
            ytrue_abs = test[abs_col].to_numpy(dtype=float)

            out_dir_h = os.path.join(out_dir, cfg["name"], name, f"h{h}")
            os.makedirs(out_dir_h, exist_ok=True)

            if h == 1:
                summary = run_grid_search(
                    model_name=name,
                    X=Xcv,
                    y=ycv,
                    cv_splits=5,
                    out_dir=out_dir_h,
                    refit_metric="r2",
                )
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
                    continue
                best_model = _fit_best_model(name, best_params, use_gpu, Xcv, ycv)
                best_model_path = os.path.join(out_dir_h, f"{name}_best.pkl")
                if name == "mlp":
                    best_model.save(best_model_path)
                else:
                    joblib.dump(best_model, best_model_path)

            yhat_target = predict_pipeline_compat(best_model, Xt)
            if USE_DELTA:
                yhat_abs = np.clip(base_t_test + yhat_target, lo_cap, hi_cap)
            else:
                yhat_abs = np.clip(yhat_target, lo_cap, hi_cap)

            abs_lo, abs_hi = _bootstrap_pred_intervals(
                name=name,
                best_params=best_params,
                use_gpu=use_gpu,
                Xcv=Xcv,
                ycv=ycv,
                ids_cv=ids_cv,
                Xt=Xt,
                base_t_test=base_t_test,
                use_delta=USE_DELTA,
            )

            abs_metrics = evaluate_regression(ytrue_abs, yhat_abs)
            naive_pred = np.clip(base_t_test, lo_cap, hi_cap)
            naive_metrics = evaluate_regression(ytrue_abs, naive_pred)

            ids_test = test[ID_COL].to_numpy()
            metric_ci = bootstrap_metric_ci_by_id(ids_test, ytrue_abs, yhat_abs, n_boot=N_BOOT, seed=BOOT_SEED, alpha=CI_ALPHA)

            print(
                f"{cfg['name']} {name} h={h} "
                f"R2={abs_metrics['R2']:.3f} "
                f"[{metric_ci['Y_R2_ci_low']:.3f},{metric_ci['Y_R2_ci_high']:.3f}] "
                f"MAE={abs_metrics['MAE']:.3f} "
                f"[{metric_ci['Y_MAE_ci_low']:.3f},{metric_ci['Y_MAE_ci_high']:.3f}] "
                f"Naive_R2={naive_metrics['R2']:.3f} "
                f"Naive_MAE={naive_metrics['MAE']:.3f}"
            )

            row = {
                "target": cfg["name"],
                "model": name,
                "h": int(h),
                "label": label_col,
                "MAE": float(abs_metrics["MAE"]),
                "RMSE": float(abs_metrics["RMSE"]),
                "R2": float(abs_metrics["R2"]),
                "R2_ci_low": metric_ci["Y_R2_ci_low"],
                "R2_ci_high": metric_ci["Y_R2_ci_high"],
                "MAE_ci_low": metric_ci["Y_MAE_ci_low"],
                "MAE_ci_high": metric_ci["Y_MAE_ci_high"],
                "Naive_MAE": float(naive_metrics["MAE"]),
                "Naive_RMSE": float(naive_metrics["RMSE"]),
                "Naive_R2": float(naive_metrics["R2"]),
                "best_model_path": best_model_path,
                "cv_results_csv": best_cv_csv,
                "n_boot": int(N_BOOT),
                "ci_alpha": float(CI_ALPHA),
                "base_col": base_col,
                "abs_col": abs_col,
            }
            all_rows.append(row)

            pred_df = pd.DataFrame(
                {
                    ID_COL: test[ID_COL].to_numpy(),
                    DAY_COL: test[DAY_COL].to_numpy(),
                    "y_true_abs": ytrue_abs,
                    "y_pred_abs": yhat_abs,
                    "y_pred_abs_ci_low": abs_lo,
                    "y_pred_abs_ci_high": abs_hi,
                    "y_naive_abs": naive_pred,
                    "base_t": base_t_test,
                }
            )
            pred_df.to_csv(os.path.join(out_dir_h, "predictions.csv"), index=False)

            if args.summarise_by in ("patient", "both"):
                sp = summarize_predictions(pred_df, ID_COL, DAY_COL, by="patient")
                sp_path = os.path.join(out_dir_h, "summary_by_patient.csv")
                sp.to_csv(sp_path, index=False)
                print(f"[SAVED] {sp_path} | n_patients={len(sp)}")
                if len(sp):
                    print(
                        f"{name} h={h} patient "
                        f"mean_R2={float(np.nanmean(sp['r2'])):.3f} "
                        f"mean_MAE={float(np.nanmean(sp['mae'])):.3f} "
                        f"n_patients={len(sp)}"
                    )
                else:
                    print(f"{name} h={h} patient summary empty (check id values / NaNs).")

            if args.summarise_by in ("day", "both"):
                sd = summarize_predictions(pred_df, ID_COL, DAY_COL, by="day")
                sd.to_csv(os.path.join(out_dir_h, "summary_by_day.csv"), index=False)

            bins_abs = _calibration_bins(ytrue_abs, yhat_abs, n_bins=N_CALIB_BINS)
            bins_abs.to_csv(os.path.join(out_dir_h, "calibration_bins_abs.csv"), index=False)
            summ_abs = _calibration_summary(ytrue_abs, yhat_abs)
            summ_abs.to_csv(os.path.join(out_dir_h, "calibration_summary_abs.csv"), index=False)

            X_for_shap = Xcv.sample(n=min(50, len(Xcv)), random_state=42).copy()
            if name == "mlp":
                shap_imp = best_model.shap_importance(X_for_shap, top_n=30)
            else:
                shap_imp = shap_importance_for_pipeline(best_model, X_for_shap, top_n=30)
            shap_imp.to_csv(os.path.join(out_dir_h, "shap_importance.csv"), header=["mean_|SHAP|"])

    summary_path = os.path.join(out_dir, f"summary_{cfg['name']}_all_models_all_horizons.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
