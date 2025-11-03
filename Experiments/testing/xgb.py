# xgb.py
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

def safe_fit(model, Xtr, ytr, Xv, yv):
    try:
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False, early_stopping_rounds=50)
        return
    except TypeError:
        pass
    try:
        from xgboost.callback import EarlyStopping
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False, callbacks=[EarlyStopping(rounds=50, save_best=True)])
        return
    except Exception:
        pass
    model.fit(Xtr, ytr)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ml-ready", required=True)
    p.add_argument("--id-col", default=None)
    p.add_argument("--day-col", default=None)
    p.add_argument("--label", default="dY_next")
    p.add_argument("--trees", type=int, default=800)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.85)
    p.add_argument("--colsample", type=float, default=0.9)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=1.2)
    args = p.parse_args()

    full = pd.read_csv(args.ml_ready)
    id_col = args.id_col or pick_col(full, "id", "ID")
    day_col = args.day_col or pick_col(full, "day", "Day")
    if id_col is None or day_col is None:
        raise ValueError("Could not resolve id/day column names")
    if args.label not in full.columns:
        raise ValueError("Label column not found")

    num_cols = full.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in {id_col, day_col, args.label}]
    df = full[[id_col, day_col] + feats + [args.label]].copy()
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").fillna(df[feats].median())

    df = split_last_frac_per_id(df, id_col, day_col, frac=0.2)
    train = df[df["__is_train"]].drop(columns="__is_train")
    test  = df[~df["__is_train"]].drop(columns="__is_train")

    Xtr, ytr = train[feats].values, train[args.label].values
    Xt, yt   = test[feats].values,  test[args.label].values

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

    n_val = max(1, int(round(0.2 * len(Xtr))))
    if n_val >= len(Xtr):
        Xv, yv = Xtr, ytr
        Xtr2, ytr2 = Xtr, ytr
    else:
        Xv, yv = Xtr[-n_val:], ytr[-n_val:]
        Xtr2, ytr2 = Xtr[:-n_val], ytr[:-n_val]

    safe_fit(model, Xtr2, ytr2, Xv, yv)

    yhat_d = model.predict(Xt)
    mae_d = mean_absolute_error(yt, yhat_d)
    r2_d = r2_score(yt, yhat_d)
    print(f"[xgb] dY_next  MAE={mae_d:.3f}  R2={r2_d:.3f}")

    if {"U_t","Ystar_next_day"}.issubset(full.columns):
        base = (
            full[[id_col, day_col, "U_t", "Ystar_next_day"]]
            .groupby([id_col, day_col], as_index=False)
            .mean(numeric_only=True)
        )
        join = test[[id_col, day_col]].merge(
            base, on=[id_col, day_col], how="left", validate="many_to_one"
        )
        ut = join["U_t"].to_numpy()
        ytrue_abs = join["Ystar_next_day"].to_numpy()
        yhat_abs = np.clip(ut + yhat_d, 0, 10)
        m = ~np.isnan(ytrue_abs)
        if m.any():
            mae_abs = mean_absolute_error(ytrue_abs[m], yhat_abs[m])
            r2_abs = r2_score(ytrue_abs[m], yhat_abs[m])
            mae_naive = mean_absolute_error(ytrue_abs[m], ut[m])
            r2_naive = r2_score(ytrue_abs[m], ut[m])
            print(f"[xgb] Y_next   MAE={mae_abs:.3f}  R2={r2_abs:.3f}")
            print(f"[naive U_t]   MAE={mae_naive:.3f} R2={r2_naive:.3f}")

    out_json = args.ml_ready.replace(".csv", "_xgb_model.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    model.save_model(out_json)
    print(out_json)

if __name__ == "__main__":
    main()
