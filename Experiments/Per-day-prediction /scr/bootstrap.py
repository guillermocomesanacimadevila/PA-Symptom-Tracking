import os
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def _bootstrap_ci(y_true, y_pred, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    mae_vals = []
    rmse_vals = []
    r2_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        mae, rmse, r2 = _compute_metrics(yt, yp)
        mae_vals.append(mae)
        rmse_vals.append(rmse)
        r2_vals.append(r2)
    mae_vals = np.array(mae_vals)
    rmse_vals = np.array(rmse_vals)
    r2_vals = np.array(r2_vals)

    def s(x):
        return float(x.mean()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    mae_mean, mae_low, mae_high = s(mae_vals)
    rmse_mean, rmse_low, rmse_high = s(rmse_vals)
    r2_mean, r2_low, r2_high = s(r2_vals)
    return {
        "MAE_boot_mean": mae_mean,
        "MAE_ci_low": mae_low,
        "MAE_ci_high": mae_high,
        "RMSE_boot_mean": rmse_mean,
        "RMSE_ci_low": rmse_low,
        "RMSE_ci_high": rmse_high,
        "R2_boot_mean": r2_mean,
        "R2_ci_low": r2_low,
        "R2_ci_high": r2_high,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", type=str, required=True)
    p.add_argument("--models", type=str, default="auto")
    p.add_argument("--h_start", type=int, default=1)
    p.add_argument("--h_end", type=int, default=14)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", type=str, required=True)
    args = p.parse_args()

    pred_dir = args.pred_dir
    if args.models == "auto":
        models = []
        for d in os.listdir(pred_dir):
            pth = os.path.join(pred_dir, d)
            if os.path.isdir(pth):
                models.append(d)
        models = sorted(models)
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = []
    for model in models:
        for h in range(args.h_start, args.h_end + 1):
            pred_path = os.path.join(pred_dir, model, f"h{h}", "predictions.csv")
            if not os.path.exists(pred_path):
                print(f"[skip] {pred_path} not found")
                continue
            df = pd.read_csv(pred_path)

            if "y_true_target" not in df.columns or "y_pred_target" not in df.columns:
                print(f"[skip] {pred_path} missing y_true_target/y_pred_target")
                continue

            y_true_target = df["y_true_target"].to_numpy(dtype=float)
            y_pred_target = df["y_pred_target"].to_numpy(dtype=float)
            if len(y_true_target) == 0:
                print(f"[skip] {pred_path} empty")
                continue

            mae_full, rmse_full, r2_full = _compute_metrics(y_true_target, y_pred_target)
            ci = _bootstrap_ci(y_true_target, y_pred_target, n_boot=args.n_boot, seed=args.seed)

            row = {
                "model": model,
                "horizon": h,
                "n": int(len(y_true_target)),
                "MAE_target_full": float(mae_full),
                "RMSE_target_full": float(rmse_full),
                "R2_target_full": float(r2_full),
                "MAE_target_boot_mean": ci["MAE_boot_mean"],
                "MAE_target_ci_low": ci["MAE_ci_low"],
                "MAE_target_ci_high": ci["MAE_ci_high"],
                "RMSE_target_boot_mean": ci["RMSE_boot_mean"],
                "RMSE_target_ci_low": ci["RMSE_ci_low"],
                "RMSE_target_ci_high": ci["RMSE_ci_high"],
                "R2_target_boot_mean": ci["R2_boot_mean"],
                "R2_target_ci_low": ci["R2_ci_low"],
                "R2_target_ci_high": ci["R2_ci_high"],
            }

            if {"y_true_abs", "y_pred_abs"}.issubset(df.columns):
                y_true_abs = df["y_true_abs"].to_numpy(dtype=float)
                y_pred_abs = df["y_pred_abs"].to_numpy(dtype=float)
                mae_a, rmse_a, r2_a = _compute_metrics(y_true_abs, y_pred_abs)
                ci_a = _bootstrap_ci(y_true_abs, y_pred_abs, n_boot=args.n_boot, seed=args.seed)
                row.update({
                    "MAE_abs_full": float(mae_a),
                    "RMSE_abs_full": float(rmse_a),
                    "R2_abs_full": float(r2_a),
                    "MAE_abs_boot_mean": ci_a["MAE_boot_mean"],
                    "MAE_abs_ci_low": ci_a["MAE_ci_low"],
                    "MAE_abs_ci_high": ci_a["MAE_ci_high"],
                    "RMSE_abs_boot_mean": ci_a["RMSE_boot_mean"],
                    "RMSE_abs_ci_low": ci_a["RMSE_ci_low"],
                    "RMSE_abs_ci_high": ci_a["RMSE_ci_high"],
                    "R2_abs_boot_mean": ci_a["R2_boot_mean"],
                    "R2_abs_ci_low": ci_a["R2_ci_low"],
                    "R2_abs_ci_high": ci_a["R2_ci_high"],
                })

                if "y_naive_abs" in df.columns:
                    y_naive_abs = df["y_naive_abs"].to_numpy(dtype=float)
                    mae_n, rmse_n, r2_n = _compute_metrics(y_true_abs, y_naive_abs)
                    ci_n = _bootstrap_ci(y_true_abs, y_naive_abs, n_boot=args.n_boot, seed=args.seed)
                    row.update({
                        "MAE_naive_abs_full": float(mae_n),
                        "RMSE_naive_abs_full": float(rmse_n),
                        "R2_naive_abs_full": float(r2_n),
                        "MAE_naive_abs_boot_mean": ci_n["MAE_boot_mean"],
                        "MAE_naive_abs_ci_low": ci_n["MAE_ci_low"],
                        "MAE_naive_abs_ci_high": ci_n["MAE_ci_high"],
                        "RMSE_naive_abs_boot_mean": ci_n["RMSE_boot_mean"],
                        "RMSE_naive_abs_ci_low": ci_n["RMSE_ci_low"],
                        "RMSE_naive_abs_ci_high": ci_n["RMSE_ci_high"],
                        "R2_naive_abs_boot_mean": ci_n["R2_boot_mean"],
                        "R2_naive_abs_ci_low": ci_n["R2_ci_low"],
                        "R2_naive_abs_ci_high": ci_n["R2_ci_high"],
                    })

            rows.append(row)
            print(f"[ok] model={model} h={h} n={len(y_true_target)} R2_target_full={r2_full:.3f}")

    if not rows:
        print("no results, nothing saved")
        return

    out_df = pd.DataFrame(rows)
    out_path = args.out_csv
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
