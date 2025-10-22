import os
import joblib
import pandas as pd
import warnings
from utils import (
    load_ml_ready,
    ensure_next_day_label,
    split_by_id_time,
    build_feature_label_matrices,
    evaluate_regression,
    run_grid_search,
)

DATA_PATH = "../Data/Symptomtrackingdata_csv-cleaned_with_vars_ml_ready.csv"
OUT_DIR = "../grid_results"
ID_COL = "id"
DAY_COL = "day"
LABEL_COL = "Ystar_next_day"
MODEL_LIST = ["lasso", "elasticnet", "rf", "svr", "knn", "xgb", "catboost", "mlp"]
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    df = load_ml_ready(DATA_PATH)
    df = ensure_next_day_label(df, id_col=ID_COL, day_col=DAY_COL, base_col="U_t", label_col=LABEL_COL)
    train, val, test = split_by_id_time(df, id_col=ID_COL, day_col=DAY_COL, test_size=0.15, val_size=0.15)
    Xtr, ytr, feat_cols = build_feature_label_matrices(train, id_col=ID_COL, day_col=DAY_COL, label_col=LABEL_COL)
    Xva, yva, _ = build_feature_label_matrices(val, id_col=ID_COL, day_col=DAY_COL, label_col=LABEL_COL)
    Xt, yt, _ = build_feature_label_matrices(test, id_col=ID_COL, day_col=DAY_COL, label_col=LABEL_COL)
    Xcv = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True)
    ycv = pd.concat([pd.Series(ytr), pd.Series(yva)], axis=0).to_numpy()
    os.makedirs(OUT_DIR, exist_ok=True)
    summaries = []
    for name in MODEL_LIST:
        print("\n==========================")
        print(f"Running model: {name.upper()}")
        print("==========================")
        try:
            summary = run_grid_search(model_name=name, X=Xcv, y=ycv, cv_splits=5, out_dir=OUT_DIR, refit_metric="r2")
        except ImportError as e:
            print(f"[{name}] skipped: {e}")
            continue
        best_path = summary["best_model_path"]
        if name == "mlp":
            from models import TorchMLPModel
            best_model = TorchMLPModel().load(best_path)
        else:
            best_model = joblib.load(best_path)
        yhat_test = best_model.predict(Xt)
        test_metrics = evaluate_regression(yt, yhat_test)
        row = {
            **summary,
            "TEST_MAE": test_metrics["MAE"],
            "TEST_RMSE": test_metrics["RMSE"],
            "TEST_R2": test_metrics["R2"],
        }
        summaries.append(row)
        print(f"[{name}] Test R²={row['TEST_R2']:.3f} | MAE={row['TEST_MAE']:.3f} | RMSE={row['TEST_RMSE']:.3f}")
    summary_path = os.path.join(OUT_DIR, "summary_all_models.csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"\n[OK] Wrote summary -> {summary_path}")

if __name__ == "__main__":
    main()
