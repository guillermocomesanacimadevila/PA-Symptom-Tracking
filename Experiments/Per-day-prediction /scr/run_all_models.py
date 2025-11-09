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
)

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = "../Data/Symptomtrackingdata_csv-cleaned_with_vars_ml_ready.csv"
OUT_DIR = "../grid_results_horizons"
ID_COL = "id"
DAY_COL = "day"
MODEL_LIST = ["lasso", "elasticnet", "rf", "svr", "knn", "xgb", "catboost", "mlp"]
H_MAX = 14
USE_DELTA = True

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

def main():
    _gpu_summary()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_ml_ready(DATA_PATH)
    all_rows = []
    for h in range(1, H_MAX + 1):
        print("=" * 40)
        print(f"HORIZON {h}")
        print("=" * 40)
        df_h, label_col = ensure_horizon_label(
            df,
            h=h,
            id_col=ID_COL,
            day_col=DAY_COL,
            base_col=BASE_COMPOSITE_COL,
            use_delta=USE_DELTA,
        )
        train, val, test = split_by_id_time(
            df_h,
            id_col=ID_COL,
            day_col=DAY_COL,
            test_size=0.15,
            val_size=0.15,
        )
        Xtr, ytr, feat_cols = build_feature_label_matrices(
            train,
            id_col=ID_COL,
            day_col=DAY_COL,
            label_col=label_col,
        )
        Xva, yva, _ = build_feature_label_matrices(
            val,
            id_col=ID_COL,
            day_col=DAY_COL,
            label_col=label_col,
        )
        Xt, yt, _ = build_feature_label_matrices(
            test,
            id_col=ID_COL,
            day_col=DAY_COL,
            label_col=label_col,
        )
        Xcv = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True)
        ycv = np.concatenate([ytr, yva])
        if BASE_COMPOSITE_COL not in test.columns:
            raise ValueError("Missing U_t on test set")
        Ut_test = test[BASE_COMPOSITE_COL].to_numpy()
        abs_col = f"Y_next_h{h}"
        if abs_col not in test.columns:
            raise ValueError(f"Missing {abs_col} on test set")
        ytrue_abs = test[abs_col].to_numpy()
        for name in MODEL_LIST:
            print("-" * 40)
            print(f"h={h} model={name}")
            print("-" * 40)
            try:
                summary = run_grid_search(
                    model_name=name,
                    X=Xcv,
                    y=ycv,
                    cv_splits=5,
                    out_dir=os.path.join(OUT_DIR, f"h{h}"),
                    refit_metric="r2",
                )
            except ImportError as e:
                print(f"h={h} {name} skipped: {e}")
                continue
            best_path = summary["best_model_path"]
            if name == "mlp":
                from models import TorchMLPModel
                best_model = TorchMLPModel().load(best_path)
            else:
                best_model = joblib.load(best_path)
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
                "h": h,
                "model": name,
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
                "best_model_path": summary["best_model_path"],
                "cv_results_csv": summary["cv_results_csv"],
            }
            all_rows.append(row)
            try:
                X_for_shap = Xcv.sample(n=min(500, len(Xcv)), random_state=42).copy()
                if name == "mlp":
                    shap_imp = best_model.shap_importance(X_for_shap, top_n=30)
                else:
                    shap_imp = shap_importance_for_pipeline(best_model, X_for_shap, top_n=30)
                imp_path = os.path.join(OUT_DIR, f"h{h}_{name}_shap_importance.csv")
                shap_imp.to_csv(imp_path, header=["mean_|SHAP|"])
                print(f"SHAP -> {imp_path}")
            except Exception as e:
                print(f"SHAP skipped: {e}")
    summary_path = os.path.join(OUT_DIR, "summary_all_models_all_horizons.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"summary -> {summary_path}")

if __name__ == "__main__":
    main()
