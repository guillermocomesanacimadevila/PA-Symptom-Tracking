import argparse
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def pick_col(df, a, b):
    if a in df.columns:
        return a
    if b in df.columns:
        return b
    return None

def split_last_frac_per_id(df, id_col, day_col, frac=0.2):
    df = df.sort_values([id_col, day_col])
    parts = []
    for _, g in df.groupby(id_col, group_keys=False):
        n = len(g)
        if n == 0:
            continue
        k = max(1, int(np.floor((1.0 - frac) * n)))
        g = g.copy()
        g["__is_train"] = False
        g.iloc[:k, g.columns.get_loc("__is_train")] = True
        parts.append(g)
    if not parts:
        raise ValueError("No data after splitting by id.")
    return pd.concat(parts, axis=0)

def get_base_features(df, id_col, day_col, dy_col):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = {id_col, day_col, dy_col}
    feats = []
    for c in num_cols:
        if c in drop:
            continue
        if c.startswith("dY_next_h") or c.startswith("Y_next_h"):
            continue
        feats.append(c)
    return feats

def filter_low_variance(X, thresh=1e-6):
    var = X.var(axis=0)
    keep = var[var >= thresh].index.tolist()
    return keep

def filter_high_corr(X, thr=0.95):
    if X.shape[1] <= 1:
        return list(X.columns)
    c = X.corr().abs()
    upper = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
    drop = set()
    for col in upper.columns:
        if (upper[col] > thr).any():
            drop.add(col)
    keep = [col for col in X.columns if col not in drop]
    return keep

def build_feature_sets(train, base_feats):
    Xtr_full = train[base_feats]

    feats_base = list(base_feats)

    lv_keep = filter_low_variance(Xtr_full, thresh=1e-6)
    feats_lv = [f for f in base_feats if f in lv_keep]

    hc_keep = filter_high_corr(Xtr_full[feats_base])
    feats_hc = [f for f in base_feats if f in hc_keep]

    if feats_lv:
        hc_keep_lv = filter_high_corr(Xtr_full[feats_lv])
        feats_lvhc = [f for f in feats_lv if f in hc_keep_lv]
    else:
        feats_lvhc = []

    configs = {
        "base": feats_base,
        "lv": feats_lv,
        "hc": feats_hc,
        "lvhc": feats_lvhc,
    }
    return configs

def run_config(df, train, test, id_col, day_col, dy_col, yabs_col, feats, cfg_name, args):
    if len(feats) == 0:
        print(f"[{cfg_name}] no features, skipping")
        return

    med = train[feats].median()
    Xtr = train[feats].fillna(med).to_numpy()
    Xt = test[feats].fillna(med).to_numpy()
    ytr = train[dy_col].to_numpy()
    yt = test[dy_col].to_numpy()

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.trees,
        max_depth=args.depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(Xtr, ytr)

    yhat_d = model.predict(Xt)
    mae_d = mean_absolute_error(yt, yhat_d)
    r2_d = r2_score(yt, yhat_d)

    txt = f"ΔY MAE={mae_d:.3f} R2={r2_d:.3f}"
    if ("U_t" in df.columns) and (yabs_col in df.columns):
        base = (
            df[[id_col, day_col, "U_t", yabs_col]]
            .groupby([id_col, day_col], as_index=False)
            .mean(numeric_only=True)
        )
        join = test[[id_col, day_col]].merge(
            base, on=[id_col, day_col], how="left"
        )
        ut = join["U_t"].to_numpy()
        ytrue_abs = join[yabs_col].to_numpy()
        if len(ytrue_abs) == len(yhat_d):
            yhat_abs = np.clip(ut + yhat_d, 0, 10)
            m = ~np.isnan(ytrue_abs)
            if m.any():
                mae_abs = mean_absolute_error(ytrue_abs[m], yhat_abs[m])
                r2_abs = r2_score(ytrue_abs[m], yhat_abs[m])
                mae_naive = mean_absolute_error(ytrue_abs[m], ut[m])
                r2_naive = r2_score(ytrue_abs[m], ut[m])
                txt += f" | Y MAE={mae_abs:.3f} R2={r2_abs:.3f} | naive R2={r2_naive:.3f}"
    print(txt)

    out_json = args.ml_ready.replace(
        ".csv", f"_xgb_h{args.h}_cfg-{cfg_name}.json"
    )
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    model.save_model(out_json)

def run_horizon(df, id_col, day_col, h, args):
    dy_col = f"dY_next_h{h}"
    yabs_col = f"Y_next_h{h}"
    if dy_col not in df.columns:
        return

    cols = [id_col, day_col, dy_col]
    if yabs_col in df.columns:
        cols.append(yabs_col)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if c not in cols:
            cols.append(c)

    dfh = df[cols].dropna(subset=[dy_col]).copy()
    if dfh.empty:
        return

    dfh = split_last_frac_per_id(dfh, id_col, day_col, frac=0.2)
    train = dfh[dfh["__is_train"]].drop(columns="__is_train")
    test = dfh[~dfh["__is_train"]].drop(columns="__is_train")

    base_feats = get_base_features(dfh, id_col, day_col, dy_col)
    cfg_feats = build_feature_sets(train, base_feats)

    print(f"[h={h}]")
    for cfg_name, feats in cfg_feats.items():
        run_config(df, train, test, id_col, day_col, dy_col, yabs_col, feats, cfg_name, args)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ml-ready", required=True)
    p.add_argument("--id-col", default=None)
    p.add_argument("--day-col", default=None)
    p.add_argument("--trees", type=int, default=800)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.85)
    p.add_argument("--colsample", type=float, default=0.9)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=1.2)
    p.add_argument("--max-h", type=int, default=7)
    args = p.parse_args()

    df = pd.read_csv(args.ml_ready)
    id_col = args.id_col or pick_col(df, "id", "ID")
    day_col = args.day_col or pick_col(df, "day", "Day")
    if id_col is None or day_col is None:
        raise ValueError("Could not resolve id/day columns.")

    for h in range(1, args.max_h + 1):
        run_horizon(df, id_col, day_col, h, args)

if __name__ == "__main__":
    main()
