import argparse
import numpy as np
import pandas as pd
from typing import List, Optional

def gb(df: pd.DataFrame, by):
    try:
        return df.groupby(by, group_keys=False, include_groups=False)
    except TypeError:
        return df.groupby(by, group_keys=False)

SYMPTOM_ITEMS_ALIASES = [
    "tiredness", "poor_concentration", "irritability", "intestinal_problems",
    "memory_loss", "muscle_pain", "nerve_pain", "pins_and_needles",
    "tinnitus", "word_finding_difficulties", "dizziness",
    "Tiredness", "Poor concentration", "Irritability", "Intestinal problems",
    "Memory loss", "Muscle pain", "Nerve pain", "Pins and needles",
    "Tinnitus", "Word-finding difficulties", "Dizziness",
]

def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def any_present(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha exponent")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta exponent")
    parser.add_argument("--use-precomputed-s", action="store_true",
                        help="Use mean_symptom_score_active (if present) as S_t")
    parser.add_argument("--id-col", default=None, help="Override participant ID column")
    parser.add_argument("--day-col", default=None, help="Override sequential day column")
    parser.add_argument("--date-col", default=None, help="Optional date/timestamp column")
    parser.add_argument("--quick-benchmark", action="store_true",
                        help="Run a quick RandomForest benchmark on the ML-ready dataset")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    ID = args.id_col or first_present(df, ["ID", "id"])
    DAY = args.day_col or first_present(df, ["Day", "day"])
    DATE = args.date_col or first_present(df, ["StartDate", "start_date"])
    FINISH = first_present(df, ["Finish", "finished"])
    TREATMENT = first_present(df, ["Treatment", "treatment"])
    TREATMENT_TYPE = first_present(df, ["TreatmentType", "treatment_type"])
    W_COL = first_present(df, ["OverallWellbeing", "overall_wellbeing"])
    S_PRE = first_present(df, ["mean_symptom_score_active", "mean_symptom_score_active"])

    TOTAL_SYM = first_present(df, ["total_symptom_score"])
    SYM_COUNT = first_present(df, ["symptom_count"])
    MAX_SYM = first_present(df, ["max_symptom_score"])
    SYM_VAR = first_present(df, ["symptom_variability"])
    COG_MEAN = first_present(df, ["cognitive_mean"])
    NEURO_MEAN = first_present(df, ["neuro_mean"])
    PAINFAT_MEAN = first_present(df, ["painfatigue_mean"])

    if ID is None or DAY is None:
        raise ValueError("Could not resolve ID or Day columns. Use --id-col/--day-col to specify.")
    if W_COL is None:
        raise ValueError("Could not find wellbeing column (OverallWellbeing / overall_wellbeing).")

    if DATE and DATE in df.columns:
        df[DATE] = pd.to_datetime(df[DATE], dayfirst=True, errors="coerce")

    df[DAY] = pd.to_numeric(df[DAY], errors="coerce")
    numeric_candidates = [W_COL, S_PRE, TOTAL_SYM, SYM_COUNT, MAX_SYM, SYM_VAR,
                          COG_MEAN, NEURO_MEAN, PAINFAT_MEAN]
    for c in numeric_candidates:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if args.use_precomputed_s and (S_PRE is not None):
        df["S_t"] = pd.to_numeric(df[S_PRE], errors="coerce")
    else:
        items_present = any_present(df, SYMPTOM_ITEMS_ALIASES)
        if not items_present:
            raise ValueError("No symptom item columns found and --use-precomputed-s not set (or column missing).")
        item_block = df[items_present].apply(pd.to_numeric, errors="coerce")
        item_block_active = item_block.replace(0, np.nan)
        df["S_t"] = item_block_active.mean(axis=1, skipna=True)

    df["W_t"] = pd.to_numeric(df[W_COL], errors="coerce").clip(0, 10)
    df["S_t"] = df["S_t"].clip(0, 10)
    df["w_t"] = df["W_t"] / 10.0
    df["q_t"] = 1.0 - (df["S_t"] / 10.0)
    alpha, beta = float(args.alpha), float(args.beta)
    df["U_t"] = 10.0 * (df["w_t"] ** alpha) * (df["q_t"] ** beta)

    sort_cols = [ID, DAY] if not (DATE and DATE in df.columns) else [ID, DAY, DATE]
    df = df.sort_values(sort_cols)
    if TREATMENT in df.columns:
        # Treatment coding: 1=treatment, 2=no treatment
        df["is_treated"] = (df[TREATMENT] == 1).astype(int)

        def _days_since_start(g: pd.DataFrame) -> pd.Series:
            out = pd.Series(0, index=g.index, dtype=float)
            if (g["is_treated"] == 1).any():
                first_day_val = g.loc[g["is_treated"] == 1, DAY].iloc[0]
                out = g[DAY] - first_day_val
                out = out.where(g["is_treated"] == 1, 0)
            return out

        df["days_since_treatment_start"] = gb(df, ID).apply(_days_since_start)
    else:
        df["is_treated"] = 0
        df["days_since_treatment_start"] = 0

    df["Ystar_next_day"] = gb(df, ID)["U_t"].shift(-1)
    out_intermediate = args.input.replace(".csv", "_with_composites.csv")
    keep_intermediate = [c for c in [
        ID, DAY, DATE, W_COL, "S_t", "w_t", "q_t", "U_t", "Ystar_next_day",
        TREATMENT, TREATMENT_TYPE, "is_treated", "days_since_treatment_start",
        TOTAL_SYM, SYM_COUNT, MAX_SYM, SYM_VAR, COG_MEAN, NEURO_MEAN, PAINFAT_MEAN
    ] if c in df.columns]
    df[keep_intermediate].to_csv(out_intermediate, index=False)

    def add_lags_rolls(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for col in ["W_t", "S_t", "U_t"]:
            g[f"{col}_lag1"] = g[col].shift(1)
            g[f"{col}_lag2"] = g[col].shift(2)
        g["dW_t"] = g["W_t"] - g["W_t_lag1"]
        g["dS_t"] = g["S_t"] - g["S_t_lag1"]
        for col in ["W_t", "S_t", "U_t"]:
            g[f"{col}_roll3_mean"] = g[col].rolling(3, min_periods=1).mean()
            g[f"{col}_roll7_mean"] = g[col].rolling(7, min_periods=1).mean()
            g[f"{col}_roll7_std"]  = g[col].rolling(7, min_periods=2).std()
        return g

    df = gb(df, ID).apply(add_lags_rolls)

    # --- Calendar / weekly cycle --- #
    # (1) 'dow' as nullable Int64 (tolerates NA in Day)
    df["dow"] = (df[DAY] % 7).astype("Int64")
    # (2) trig features: fill NA with 0 for sin/cos calc only
    _dow_float = df["dow"].astype("float64").fillna(0.0)
    df["dow_sin"] = np.sin(2 * np.pi * _dow_float / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * _dow_float / 7.0)
    df["days_since_start"] = gb(df, ID)[DAY].transform(lambda s: s - s.min())
    data = df.dropna(subset=["Ystar_next_day"]).copy()
    base_features = [
        "W_t", "S_t", "U_t",
        TOTAL_SYM, SYM_COUNT, MAX_SYM, SYM_VAR,
        COG_MEAN, NEURO_MEAN, PAINFAT_MEAN,
        "is_treated", TREATMENT_TYPE,
        DAY, "days_since_start", "days_since_treatment_start",
        "dow", "dow_sin", "dow_cos",
    ]
    base_features = [c for c in base_features if (c is not None) and (c in data.columns)]
    derived_features = [
        "W_t_lag1", "S_t_lag1", "U_t_lag1",
        "W_t_lag2", "S_t_lag2", "U_t_lag2",
        "dW_t", "dS_t",
        "W_t_roll3_mean", "S_t_roll3_mean", "U_t_roll3_mean",
        "W_t_roll7_mean", "S_t_roll7_mean", "U_t_roll7_mean",
        "S_t_roll7_std", "U_t_roll7_std",
    ]
    derived_features = [c for c in derived_features if c in data.columns]

    features = base_features + derived_features
    label = "Ystar_next_day"
    ml_cols = [ID, DAY] + features + [label]
    ml_cols = [c for c in ml_cols if c in data.columns]
    out_ml = args.input.replace(".csv", "_ml_ready.csv")
    data[ml_cols].to_csv(out_ml, index=False)

    print(f"[OK] Wrote intermediate with composites: {out_intermediate}")
    print(f"[OK] Wrote ML-ready dataset:            {out_ml}")
    print(f"[INFO] Rows with label: {len(data)} | Participants: {data[ID].nunique()}")
    na_rate = data[features].isna().mean().sort_values(ascending=False)
    print("[INFO] Top 10 feature NA rates:")
    print(na_rate.head(10))

    if args.quick_benchmark:
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import mean_absolute_error, r2_score
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance
        except Exception as e:
            print("[WARN] Skipping quick benchmark (scikit-learn not available):", e)
            return

        def time_split_mask(group, frac_val=0.2):
            n = len(group)
            split = int(np.floor((1 - frac_val) * n))
            return pd.Series([True]*split + [False]*(n - split), index=group.index)

        mask = gb(data, ID).apply(time_split_mask)
        X, y = data[features].copy(), data[label].astype(float)

        cat_features = []
        if TREATMENT_TYPE and (TREATMENT_TYPE in X.columns):
            cat_features.append(TREATMENT_TYPE)
        num_features = [c for c in features if c not in cat_features]

        numeric_t = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True))
        ])
        categorical_t = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preproc = ColumnTransformer(
            [("num", numeric_t, num_features),
             ("cat", categorical_t, cat_features)],
            remainder="drop"
        )

        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        pipe = Pipeline([("prep", preproc), ("model", model)])

        X_train, X_valid = X[mask], X[~mask]
        y_train, y_valid = y[mask], y[~mask]

        pipe.fit(X_train, y_train)
        y_pred = np.clip(pipe.predict(X_valid), 0, 10)

        mae = mean_absolute_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)
        print(f"[RF] Validation MAE: {mae:.3f} | R^2: {r2:.3f}")

        try:
            imp = permutation_importance(pipe, X_valid, y_valid, n_repeats=10, random_state=42)
            cat_names = []
            if cat_features:
                ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
                cat_names = list(ohe.get_feature_names_out(cat_features))
            all_names = num_features + cat_names
            order = np.argsort(imp.importances_mean)[::-1][:20]
            print("[RF] Top features:")
            for idx in order:
                print(f"{all_names[idx]:35s}  {imp.importances_mean[idx]:.4f} ± {imp.importances_std[idx]:.4f}")
        except Exception as e:
            print("[WARN] Permutation importance skipped:", e)

if __name__ == "__main__":
    main()