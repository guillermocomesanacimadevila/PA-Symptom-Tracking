import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.base import clone
from tqdm import tqdm
from utils import (
    load_ml_ready,
    build_treated_events,
    build_control_events,
    compute_event_weights_by_id,
    make_pipeline_and_grid,
    evaluate_regression,
    predict_pipeline_compat,
    shap_importance_for_pipeline,
    tqdm_joblib,
)
import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning, UndefinedMetricWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def split_events_by_id(df, id_col="id", test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    ids = df[id_col].dropna().unique()
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(test_size * n)))
    test_ids = set(ids[:n_test])
    train_ids = set(ids[n_test:])
    train = df[df[id_col].isin(train_ids)].copy()
    test = df[df[id_col].isin(test_ids)].copy()
    return train, test

def prepare_arrays(df, feat_cols, id_col):
    X = df[feat_cols].copy()
    y = df["Y_star"].to_numpy()
    a = df["A_star"].to_numpy()
    ids = df[id_col].to_numpy()
    w = compute_event_weights_by_id(df, id_col=id_col)
    return X, y, a, ids, w

def parse_model_list(s):
    if s is None or s.strip() == "":
        return ["lasso", "elasticnet", "rf", "svr", "knn", "xgb", "catboost", "mlp"]
    parts = [x.strip() for x in s.split(",")]
    parts = [x for x in parts if x != ""]
    if len(parts) == 0:
        return ["lasso", "elasticnet", "rf", "svr", "knn", "xgb", "catboost", "mlp"]
    return parts

def cluster_bootstrap_ate(out_pred, id_col, B=300, seed=42):
    rng = np.random.default_rng(seed)
    ids = out_pred[id_col].dropna().unique()
    ate_vals = []
    ate_dr_vals = []
    for _ in range(B):
        bs_ids = rng.choice(ids, size=len(ids), replace=True)
        parts = []
        for pid in bs_ids:
            parts.append(out_pred[out_pred[id_col] == pid])
        bs = pd.concat(parts, ignore_index=True)
        tau = bs["tau_hat"].to_numpy()
        tau_dr = bs["tau_hat_dr"].to_numpy()
        ate_vals.append(float(np.mean(tau)))
        ate_dr_vals.append(float(np.mean(tau_dr)))
    return np.array(ate_vals), np.array(ate_dr_vals)

def crossfit_outcomes(model_treated, model_control, X, y, a, groups, n_splits=5):
    cv = GroupKFold(n_splits=n_splits)
    mu1 = np.full(len(X), np.nan, dtype=float)
    mu0 = np.full(len(X), np.nan, dtype=float)
    for tr_idx, va_idx in cv.split(X, groups=groups):
        tr_t = tr_idx[a[tr_idx] == 1]
        tr_c = tr_idx[a[tr_idx] == 0]
        m1 = clone(model_treated)
        m0 = clone(model_control)
        if len(tr_t) > 0:
            m1.fit(X.iloc[tr_t], y[tr_t])
            mu1[va_idx] = predict_pipeline_compat(m1, X.iloc[va_idx])
        if len(tr_c) > 0:
            m0.fit(X.iloc[tr_c], y[tr_c])
            mu0[va_idx] = predict_pipeline_compat(m0, X.iloc[va_idx])
    return mu1, mu0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--id_col", default="id")
    p.add_argument("--day_col", default="day")
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--H", type=int, default=3)
    p.add_argument("--outcome_col", default="U_t")
    p.add_argument("--models", default="lasso,elasticnet,rf,svr,knn,xgb,catboost,mlp")
    p.add_argument("--max_controls_per_id", type=int, default=None)
    p.add_argument("--test_size", type=float, default=0.2)
    args = p.parse_args()

    df = load_ml_ready(args.data)
    print("[OK] Loaded dataset: %s | shape=%s" % (args.data, df.shape))

    treated = build_treated_events(
        df,
        L=args.L,
        H=args.H,
        id_col=args.id_col,
        day_col=args.day_col,
        treat_col="is_treated",
        treat_type_col="treatment_type",
        outcome_col=args.outcome_col,
        inj_codes=(1, 2),
    )
    control = build_control_events(
        df,
        L=args.L,
        H=args.H,
        id_col=args.id_col,
        day_col=args.day_col,
        treat_col="is_treated",
        outcome_col=args.outcome_col,
        max_per_id=args.max_controls_per_id,
    )

    events = pd.concat([treated, control], ignore_index=True)
    events = events.dropna(subset=["Y_star"])
    events = events.dropna(axis=1, how="all")

    print("[EVENTS] rows=%d treated=%d control=%d" % (len(events), (events["A_star"] == 1).sum(), (events["A_star"] == 0).sum()))

    num_cols = events.select_dtypes(include=[np.number]).columns.tolist()
    drop = {args.id_col, "centre_day", "A_star", "Y_star"}

    leak_exact = {
        "U_minus_person_mean",
        "person_mean",
    }

    leak_prefixes = (
        "Y_next_h",
        "dY_next_h",
        "Ystar_next_day",
        "dY_next",
    )

    feat_cols = []
    for c in num_cols:
        if c in drop:
            continue
        if c in leak_exact:
            continue
        if any(c.startswith(p) for p in leak_prefixes):
            continue
        feat_cols.append(c)

    print("[FEATS] n_features=%d" % len(feat_cols))

    train_df, test_df = split_events_by_id(events, id_col=args.id_col, test_size=args.test_size, seed=42)
    print("[SPLIT] train_rows=%d test_rows=%d" % (len(train_df), len(test_df)))

    X_train, y_train, a_train, ids_train, w_train = prepare_arrays(train_df, feat_cols, args.id_col)
    X_test, y_test, a_test, ids_test, w_test = prepare_arrays(test_df, feat_cols, args.id_col)

    cv_prop = GroupKFold(n_splits=5)
    penalty_settings = [
        {"name": "unpenalized", "penalty": None, "solver": "lbfgs"},
        {"name": "l1", "penalty": "l1", "solver": "liblinear"},
        {"name": "l2", "penalty": "l2", "solver": "lbfgs"},
    ]

    rows = []
    for cfg in tqdm(penalty_settings, desc="Propensity penalty search"):
        fold_losses = []
        for tr_idx, va_idx in cv_prop.split(X_train, groups=ids_train):
            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            a_tr = a_train[tr_idx]
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(
                    penalty=cfg["penalty"],
                    solver=cfg["solver"],
                    l1_ratio=None,
                    max_iter=2000,
                )),
            ])
            pipe.fit(X_tr, a_tr)
            ps_va = pipe.predict_proba(X_va)[:, 1]
            loss = log_loss(a_train[va_idx], ps_va)
            fold_losses.append(loss)
        mean_loss = float(np.mean(fold_losses))
        rows.append({
            "name": cfg["name"],
            "penalty": cfg["penalty"],
            "solver": cfg["solver"],
            "mean_log_loss": mean_loss,
        })
        print("prop_spec:", cfg["name"], "mean_log_loss=", mean_loss)

    df_losses = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "propensity_logit_losses.csv")
    df_losses.to_csv(csv_path, index=False)
    print("Propensity losses saved to:", csv_path)
    print(df_losses)

    best = min(rows, key=lambda r: r["mean_log_loss"])
    print("Best propensity spec:", best)

    prop_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            penalty=best["penalty"],
            solver=best["solver"],
            l1_ratio=None,
            max_iter=2000,
        )),
    ])

    ps_train = np.zeros(len(X_train), dtype=float)
    for tr_idx, va_idx in tqdm(list(cv_prop.split(X_train, groups=ids_train)), desc="Propensity CV best"):
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        a_tr = a_train[tr_idx]
        prop_pipe.fit(X_tr, a_tr)
        ps_va = prop_pipe.predict_proba(X_va)[:, 1]
        ps_train[va_idx] = ps_va

    eps = 1e-3
    ps_train = np.clip(ps_train, eps, 1.0 - eps)
    prop_pipe.fit(X_train, a_train)
    ps_test = prop_pipe.predict_proba(X_test)[:, 1]
    ps_test = np.clip(ps_test, eps, 1.0 - eps)

    model_names = parse_model_list(args.models)
    summary_rows = []

    for name in model_names:
        out_dir_m = os.path.join(args.out_dir, name)
        os.makedirs(out_dir_m, exist_ok=True)

        pipe, grid = make_pipeline_and_grid(name, use_gpu=False)
        cv = GroupKFold(n_splits=5)

        mask_tr_treated = a_train == 1
        mask_tr_control = a_train == 0

        X_tr_treated = X_train[mask_tr_treated]
        y_tr_treated = y_train[mask_tr_treated]
        g_tr_treated = ids_train[mask_tr_treated]

        X_tr_control = X_train[mask_tr_control]
        y_tr_control = y_train[mask_tr_control]
        g_tr_control = ids_train[mask_tr_control]

        n_cand = len(ParameterGrid(grid))
        n_splits = cv.get_n_splits()

        gs_treated = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="r2",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        with tqdm_joblib(tqdm(total=n_cand * n_splits, desc=f"{name} treated CV", unit="fit")):
            gs_treated.fit(X_tr_treated, y_tr_treated, groups=g_tr_treated)
        model_treated = gs_treated.best_estimator_

        gs_control = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="r2",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        with tqdm_joblib(tqdm(total=n_cand * n_splits, desc=f"{name} control CV", unit="fit")):
            gs_control.fit(X_tr_control, y_tr_control, groups=g_tr_control)
        model_control = gs_control.best_estimator_

        joblib.dump(model_treated, os.path.join(out_dir_m, "model_treated.pkl"))
        joblib.dump(model_control, os.path.join(out_dir_m, "model_control.pkl"))

        y1 = predict_pipeline_compat(model_treated, X_test)
        y0 = predict_pipeline_compat(model_control, X_test)
        tau = y1 - y0

        ate = float(np.mean(tau))
        ate_w = float(np.sum(tau * w_test) / np.sum(w_test))

        y1_train_cf, y0_train_cf = crossfit_outcomes(
            model_treated=model_treated,
            model_control=model_control,
            X=X_train,
            y=y_train,
            a=a_train,
            groups=ids_train,
            n_splits=5
        )

        A_tr = a_train.astype(float)
        Y_tr = y_train.astype(float)
        tau_dr_train = (
            (A_tr / ps_train) * (Y_tr - y1_train_cf)
            - ((1.0 - A_tr) / (1.0 - ps_train)) * (Y_tr - y0_train_cf)
            + y1_train_cf - y0_train_cf
        )

        te_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
        te_pipe.fit(X_train, tau_dr_train)
        tau_dr_test = te_pipe.predict(X_test)

        ate_dr = float(np.mean(tau_dr_test))
        ate_dr_w = float(np.sum(tau_dr_test * w_test) / np.sum(w_test))

        yhat_obs_test = a_test * y1 + (1.0 - a_test) * y0
        m_overall = evaluate_regression(y_test, yhat_obs_test)

        out_pred = test_df.copy()
        out_pred["tau_hat"] = tau
        out_pred["tau_hat_dr"] = tau_dr_test
        out_pred["Yhat_treated"] = y1
        out_pred["Yhat_control"] = y0
        out_pred["Yhat_obs"] = yhat_obs_test

        ate_bs, ate_dr_bs = cluster_bootstrap_ate(out_pred, args.id_col, B=300, seed=42)
        ate_ci_low, ate_ci_high = np.quantile(ate_bs, [0.025, 0.975])
        ate_dr_ci_low, ate_dr_ci_high = np.quantile(ate_dr_bs, [0.025, 0.975])

        bs_df = pd.DataFrame({
            "ATE_bs": ate_bs,
            "ATE_DR_bs": ate_dr_bs,
        })
        bs_df.to_csv(os.path.join(out_dir_m, "bootstrap_ate_samples.csv"), index=False)

        per_patient_rows = []
        for pid, g in out_pred.groupby(args.id_col):
            y_true_p = g["Y_star"].to_numpy()
            yhat_p = g["Yhat_obs"].to_numpy()
            met = evaluate_regression(y_true_p, yhat_p)
            tau_p = g["tau_hat"].to_numpy()
            tau_dr_p = g["tau_hat_dr"].to_numpy()
            per_patient_rows.append({
                args.id_col: pid,
                "MAE_obs": met["MAE"],
                "RMSE_obs": met["RMSE"],
                "R2_obs": met["R2"],
                "tau_hat_mean": float(np.mean(tau_p)),
                "tau_hat_dr_mean": float(np.mean(tau_dr_p)),
            })

        per_patient_df = pd.DataFrame(per_patient_rows)
        per_patient_df.to_csv(os.path.join(out_dir_m, "per_patient_summary.csv"), index=False)

        shap_treated = shap_importance_for_pipeline(model_treated, X_tr_treated, top_n=50)
        shap_control = shap_importance_for_pipeline(model_control, X_tr_control, top_n=50)

        shap_treated.to_csv(os.path.join(out_dir_m, "shap_importance_treated.csv"))
        shap_control.to_csv(os.path.join(out_dir_m, "shap_importance_control.csv"))

        print("MODEL %s" % name)
        print("ATE=%.4f ATE_w=%.4f ATE_ci=[%.4f, %.4f]" % (ate, ate_w, ate_ci_low, ate_ci_high))
        print("ATE_DR=%.4f ATE_DR_w=%.4f ATE_DR_ci=[%.4f, %.4f]" % (ate_dr, ate_dr_w, ate_dr_ci_low, ate_dr_ci_high))
        print("Fit overall R2=%.3f" % m_overall["R2"])

        summary_rows.append({
            "model": name,
            "ATE": ate,
            "ATE_weighted": ate_w,
            "ATE_ci_low": float(ate_ci_low),
            "ATE_ci_high": float(ate_ci_high),
            "ATE_DR": ate_dr,
            "ATE_DR_weighted": ate_dr_w,
            "ATE_DR_ci_low": float(ate_dr_ci_low),
            "ATE_DR_ci_high": float(ate_dr_ci_high),
            "fit_overall_R2": m_overall["R2"],
        })

    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(args.out_dir, "t_learner_summary.csv"), index=False)

if __name__ == "__main__":
    main()
