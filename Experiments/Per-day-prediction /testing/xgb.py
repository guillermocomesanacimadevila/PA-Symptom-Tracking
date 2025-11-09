import argparse, os, numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def pick_col(df, a, b):
    return a if a in df.columns else (b if b in df.columns else None)

def split_last_frac_per_id(df, id_col, day_col, frac=0.2):
    df = df.sort_values([id_col, day_col])
    parts = []
    for _, g in df.groupby(id_col, group_keys=False):
        n = len(g)
        k = max(1, int(np.floor((1 - frac) * n)))
        g = g.copy()
        g["__is_train"] = False
        g.iloc[:k, g.columns.get_loc("__is_train")] = True
        parts.append(g)
    return pd.concat(parts, axis=0)

def run_horizon(df, id_col, day_col, h, args):
    dy_col = f"dY_next_h{h}"
    yabs_col = f"Y_next_h{h}"
    if dy_col not in df.columns or yabs_col not in df.columns:
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in {id_col, day_col, dy_col} and not c.startswith("dY_next_h") and not c.startswith("Y_next_h")]
    dfh = df[[id_col, day_col] + feats + [dy_col]].dropna(subset=[dy_col]).copy()
    med = dfh[feats].median()
    dfh[feats] = dfh[feats].fillna(med)

    dfh = split_last_frac_per_id(dfh, id_col, day_col, frac=0.2)
    train = dfh[dfh["__is_train"]].drop(columns="__is_train")
    test  = dfh[~dfh["__is_train"]].drop(columns="__is_train")

    Xtr, ytr = train[feats].values, train[dy_col].values
    Xt, yt   = test[feats].values,  test[dy_col].values

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

    if {"U_t", yabs_col}.issubset(df.columns):
        base = df[[id_col, day_col, "U_t", yabs_col]].groupby([id_col, day_col], as_index=False).mean(numeric_only=True)
        join = test[[id_col, day_col]].merge(base, on=[id_col, day_col], how="left")
        ut = join["U_t"].to_numpy()
        ytrue_abs = join[yabs_col].to_numpy()
        yhat_abs = np.clip(ut + yhat_d, 0, 10)
        m = ~np.isnan(ytrue_abs)
        if m.any():
            mae_abs = mean_absolute_error(ytrue_abs[m], yhat_abs[m])
            r2_abs = r2_score(ytrue_abs[m], yhat_abs[m])
            mae_naive = mean_absolute_error(ytrue_abs[m], ut[m])
            r2_naive = r2_score(ytrue_abs[m], ut[m])
            print(f"[h={h}] ΔY MAE={mae_d:.3f} R²={r2_d:.3f} | Y MAE={mae_abs:.3f} R²={r2_abs:.3f} | naive R²={r2_naive:.3f}")
        else:
            print(f"[h={h}] ΔY MAE={mae_d:.3f} R²={r2_d:.3f}")
    else:
        print(f"[h={h}] ΔY MAE={mae_d:.3f} R²={r2_d:.3f}")

    out_json = args.ml_ready.replace(".csv", f"_xgb_h{h}.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    model.save_model(out_json)

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
    for h in range(1, args.max_h + 1):
        run_horizon(df, id_col, day_col, h, args)

if __name__ == "__main__":
    main()
