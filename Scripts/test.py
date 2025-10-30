import os, warnings, joblib, pandas as pd
from tqdm import tqdm
from utils import (
    load_ml_ready,
    ensure_next_day_label,
    split_by_id_time,
    build_feature_label_matrices,
    evaluate_regression,
    shap_importance_for_pipeline,
    gpu_available,
    predict_pipeline_compat,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

DATA_PATH = "../Data/Symptomtrackingdata_csv-cleaned_with_vars_ml_ready.csv"
OUT_DIR = "../quick_results"
ID_COL = "id"
DAY_COL = "day"
LABEL_COL = "Ystar_next_day"

os.makedirs(OUT_DIR, exist_ok=True)

df = load_ml_ready(DATA_PATH)
df = ensure_next_day_label(df, id_col=ID_COL, day_col=DAY_COL, base_col="U_t", label_col=LABEL_COL)
train, val, test = split_by_id_time(df, id_col=ID_COL, day_col=DAY_COL, test_size=0.15, val_size=0.15)

Xtr, ytr, _ = build_feature_label_matrices(train, ID_COL, DAY_COL, LABEL_COL)
Xva, yva, _ = build_feature_label_matrices(val, ID_COL, DAY_COL, LABEL_COL)
Xt, yt, _ = build_feature_label_matrices(test, ID_COL, DAY_COL, LABEL_COL)

Xcv = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True)
ycv = pd.concat([pd.Series(ytr), pd.Series(yva)], axis=0).to_numpy()

use_gpu = gpu_available()

models = {
    "lasso": Pipeline([
        ("imp", SimpleImputer()),
        ("sc", StandardScaler()),
        ("m", Lasso(alpha=0.03, max_iter=6000)),
    ]),
    "rf": Pipeline([
        ("imp", SimpleImputer()),
        ("m", RandomForestRegressor(
            n_estimators=80,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ]),
    "svr": Pipeline([
        ("imp", SimpleImputer()),
        ("sc", StandardScaler()),
        ("m", SVR(C=1.0, epsilon=0.2, gamma="scale")),
    ]),
    "knn": Pipeline([
        ("imp", SimpleImputer()),
        ("sc", StandardScaler()),
        ("m", KNeighborsRegressor(n_neighbors=12, weights="distance", n_jobs=-1)),
    ]),
    "xgb": Pipeline([
        ("imp", SimpleImputer()),
        ("m", XGBRegressor(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            tree_method="gpu_hist" if use_gpu else "hist",
            predictor="gpu_predictor" if use_gpu else "auto",
            n_jobs=-1,
            objective="reg:squarederror",
        )),
    ]),
    "catboost": Pipeline([
        ("imp", SimpleImputer()),
        ("m", CatBoostRegressor(
            iterations=240,
            depth=6,
            learning_rate=0.07,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            task_type="GPU" if use_gpu else "CPU",
        )),
    ]),
}

rows = []

for name in tqdm(models, desc="models", ncols=70):
    model = models[name]
    model.fit(Xcv, ycv)
    yhat = predict_pipeline_compat(model, Xt)
    mets = evaluate_regression(yt, yhat)
    rows.append({"model": name, "MAE": mets["MAE"], "RMSE": mets["RMSE"], "R2": mets["R2"]})
    joblib.dump(model, os.path.join(OUT_DIR, f"{name}.pkl"))

    if name in ("svr", "knn"):
        xs = Xcv.sample(n=min(15, len(Xcv)), random_state=42)
    else:
        xs = Xcv.sample(n=min(40, len(Xcv)), random_state=42)

    shap_all = shap_importance_for_pipeline(model, xs, top_n=xs.shape[1])
    shap_all.to_csv(os.path.join(OUT_DIR, f"{name}_shap.csv"), header=["mean_|SHAP|"])

    print(f"\n{name.upper()} | MAE={mets['MAE']:.3f} RMSE={mets['RMSE']:.3f} R2={mets['R2']:.3f}")
    print(shap_all)

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
print("\nFINAL SUMMARY")
print(summary)
