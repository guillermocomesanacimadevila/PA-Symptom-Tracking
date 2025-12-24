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
    "tiredness","poor_concentration","irritability","intestinal_problems",
    "memory_loss","muscle_pain","nerve_pain","pins_and_needles",
    "tinnitus","word_finding_difficulties","dizziness",
    "Tiredness","Poor concentration","Irritability","Intestinal problems",
    "Memory loss","Muscle pain","Nerve pain","Pins and needles",
    "Tinnitus","Word-finding difficulties","Dizziness",
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--use-precomputed-s", action="store_true")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--day-col", default=None)
    parser.add_argument("--date-col", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    ID = args.id_col or first_present(df, ["ID","id"])
    DAY = args.day_col or first_present(df, ["Day","day"])
    DATE = args.date_col or first_present(df, ["StartDate","start_date"])
    TREATMENT = first_present(df, ["Treatment","treatment"])
    TREATMENT_TYPE = first_present(df, ["TreatmentType","treatment_type"])
    W_COL = first_present(df, ["OverallWellbeing","overall_wellbeing"])
    S_PRE = first_present(df, ["mean_symptom_score_active","mean_symptom_score_active"])
    TOTAL_SYM = first_present(df, ["total_symptom_score"])
    SYM_COUNT = first_present(df, ["symptom_count"])
    MAX_SYM = first_present(df, ["max_symptom_score"])
    SYM_VAR = first_present(df, ["symptom_variability"])
    COG_MEAN = first_present(df, ["cognitive_mean"])
    NEURO_MEAN = first_present(df, ["neuro_mean"])
    PAINFAT_MEAN = first_present(df, ["painfatigue_mean"])

    if ID is None or DAY is None:
        raise ValueError("Missing ID or Day")
    if W_COL is None:
        raise ValueError("Missing wellbeing column")

    if DATE and DATE in df.columns:
        df[DATE] = pd.to_datetime(df[DATE], dayfirst=True, errors="coerce")

    df[DAY] = pd.to_numeric(df[DAY], errors="coerce")
    for c in [W_COL,S_PRE,TOTAL_SYM,SYM_COUNT,MAX_SYM,SYM_VAR,COG_MEAN,NEURO_MEAN,PAINFAT_MEAN]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if args.use_precomputed_s and (S_PRE is not None):
        df["S_t"] = pd.to_numeric(df[S_PRE], errors="coerce")
    else:
        items_present = any_present(df, SYMPTOM_ITEMS_ALIASES)
        if not items_present:
            raise ValueError("No symptom items and --use-precomputed-s not set")
        item_block = df[items_present].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
        df["S_t"] = item_block.mean(axis=1, skipna=True)

    df["W_t"] = pd.to_numeric(df[W_COL], errors="coerce").clip(0,10)
    df["S_t"] = df["S_t"].clip(0,10)
    df["w_t"] = df["W_t"]/10.0
    df["q_t"] = 1.0 - (df["S_t"]/10.0)
    alpha, beta = float(args.alpha), float(args.beta)
    df["U_t"] = 10.0*(df["w_t"]**alpha)*(df["q_t"]**beta)
    df["wq"] = df["w_t"]*df["q_t"]
    df["w_t2"] = df["w_t"]**2
    df["q_t2"] = df["q_t"]**2

    sort_cols = [ID,DAY] if not (DATE and DATE in df.columns) else [ID,DAY,DATE]
    df = df.sort_values(sort_cols)

    if TREATMENT in df.columns:
        df["is_treated"] = (df[TREATMENT] == 1).astype(int)
        def _days_since_start(g: pd.DataFrame) -> pd.Series:
            out = pd.Series(0.0, index=g.index)
            if (g["is_treated"] == 1).any():
                t0 = g.loc[g["is_treated"] == 1, DAY].iloc[0]
                out = (g[DAY] - t0).clip(lower=0)
            return out
        df["days_since_treatment_start"] = gb(df, ID).apply(_days_since_start)
    else:
        df["is_treated"] = 0
        df["days_since_treatment_start"] = 0.0

    df["treated_today"] = df["is_treated"]
    df["treated_yday"] = gb(df, ID)["is_treated"].shift(1).fillna(0).astype(int)
    treated_last3 = gb(df, ID)["is_treated"].apply(lambda s: s.rolling(3, min_periods=1).max())
    df["treated_last3_any"] = treated_last3.reset_index(level=0, drop=True).fillna(0).astype(int)
    def _since_last_treat(s: pd.Series) -> pd.Series:
        c = s.eq(1).cumsum()
        out = (~s.eq(1)).groupby(c).cumcount()
        return out.where(c>0, np.nan).fillna(1e9)
    df["days_since_last_treat"] = gb(df, ID)["is_treated"].apply(_since_last_treat).reset_index(level=0, drop=True)

    for h in range(1, 15):
        df[f"Y_next_h{h}"] = gb(df, ID)["U_t"].shift(-h)
        df[f"dY_next_h{h}"] = df[f"Y_next_h{h}"] - df["U_t"]
    df["Ystar_next_day"] = df["Y_next_h1"]
    df["dY_next"] = df["dY_next_h1"]

    out_intermediate = args.input.replace(".csv", "_with_composites.csv")
    keep_intermediate = [c for c in [
        ID,DAY,DATE,W_COL,"S_t","w_t","q_t","U_t","Ystar_next_day","dY_next",
        "Y_next_h1","Y_next_h2","Y_next_h3","Y_next_h4","Y_next_h5","Y_next_h6","Y_next_h7",
        "Y_next_h8","Y_next_h9","Y_next_h10","Y_next_h11","Y_next_h12","Y_next_h13","Y_next_h14",
        "dY_next_h1","dY_next_h2","dY_next_h3","dY_next_h4","dY_next_h5","dY_next_h6","dY_next_h7",
        "dY_next_h8","dY_next_h9","dY_next_h10","dY_next_h11","dY_next_h12","dY_next_h13","dY_next_h14",
        TREATMENT,TREATMENT_TYPE,"is_treated","days_since_treatment_start",
        "treated_today","treated_yday","treated_last3_any","days_since_last_treat",
        TOTAL_SYM,SYM_COUNT,MAX_SYM,SYM_VAR,COG_MEAN,NEURO_MEAN,PAINFAT_MEAN
    ] if c in df.columns]
    df[keep_intermediate].to_csv(out_intermediate, index=False)

    def add_lags_rolls(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for col in ["W_t","S_t","U_t"]:
            g[f"{col}_lag1"] = g[col].shift(1)
            g[f"{col}_lag2"] = g[col].shift(2)
            g[f"{col}_lag7"] = g[col].shift(7)
            g[f"{col}_roll3_mean"] = g[col].rolling(3, min_periods=1).mean()
            g[f"{col}_roll7_mean"] = g[col].rolling(7, min_periods=1).mean()
            g[f"{col}_roll7_std"] = g[col].rolling(7, min_periods=2).std()
            g[f"{col}_roll14_mean"] = g[col].rolling(14, min_periods=3).mean()
            g[f"{col}_roll14_std"] = g[col].rolling(14, min_periods=3).std()
            g[f"{col}_ewm_a03"] = g[col].shift(1).ewm(alpha=0.30, adjust=False).mean()
            g[f"{col}_ewm_a10"] = g[col].shift(1).ewm(alpha=0.10, adjust=False).mean()
        g["dW_t"] = g["W_t"] - g["W_t_lag1"]
        g["dS_t"] = g["S_t"] - g["S_t_lag1"]
        g["dU_t"] = g["U_t"] - g["U_t_lag1"]
        return g

    df = gb(df, ID).apply(add_lags_rolls)

    if DATE and (DATE in df.columns):
        df["dow"] = df[DATE].dt.dayofweek.astype("Int64")
    else:
        df["dow"] = (df[DAY] % 7).astype("Int64")
    _dow = df["dow"].astype(float).fillna(0.0)
    df["dow_sin"] = np.sin(2*np.pi*_dow/7.0)
    df["dow_cos"] = np.cos(2*np.pi*_dow/7.0)
    df["days_since_start"] = gb(df, ID)[DAY].transform(lambda s: s - s.min())

    df["U_runmean_14"] = gb(df, ID)["U_t"].transform(lambda s: s.shift(1).rolling(14, min_periods=3).mean())
    df["U_runstd_14"] = gb(df, ID)["U_t"].transform(lambda s: s.shift(1).rolling(14, min_periods=3).std())
    exp_mean = gb(df, ID)["U_t"].transform(lambda s: s.shift(1).expanding().mean())
    exp_std = gb(df, ID)["U_t"].transform(lambda s: s.shift(1).expanding().std())
    df["U_within_z"] = (df["U_t"] - exp_mean) / exp_std.replace(0, np.nan)
    person_mean = gb(df, ID)["U_t"].transform("mean")
    df["U_minus_person_mean"] = df["U_t"] - person_mean

    label_cols = [f"dY_next_h{i}" for i in range(1,15)] + [f"Y_next_h{i}" for i in range(1,15)] + ["dY_next","Ystar_next_day"]
    base_features = [
        "W_t","S_t","U_t","w_t","q_t","wq","w_t2","q_t2",
        TOTAL_SYM,SYM_COUNT,MAX_SYM,SYM_VAR,COG_MEAN,NEURO_MEAN,PAINFAT_MEAN,
        "is_treated",TREATMENT_TYPE,
        DAY,"days_since_start","days_since_treatment_start",
        "treated_today","treated_yday","treated_last3_any","days_since_last_treat",
        "dow","dow_sin","dow_cos",
        "W_t_lag1","S_t_lag1","U_t_lag1",
        "W_t_lag2","S_t_lag2","U_t_lag2",
        "W_t_lag7","S_t_lag7","U_t_lag7",
        "dW_t","dS_t","dU_t",
        "W_t_roll3_mean","S_t_roll3_mean","U_t_roll3_mean",
        "W_t_roll7_mean","S_t_roll7_mean","U_t_roll7_mean",
        "W_t_roll7_std","S_t_roll7_std","U_t_roll7_std",
        "W_t_roll14_mean","S_t_roll14_mean","U_t_roll14_mean",
        "W_t_roll14_std","S_t_roll14_std","U_t_roll14_std",
        "W_t_ewm_a03","S_t_ewm_a03","U_t_ewm_a03",
        "W_t_ewm_a10","S_t_ewm_a10","U_t_ewm_a10",
        "U_runmean_14","U_runstd_14","U_within_z","U_minus_person_mean",
    ]
    base_features = [c for c in base_features if (c is not None) and (c in df.columns)]
    ml_cols = [ID,DAY] + base_features + [c for c in label_cols if c in df.columns]
    out_ml = args.input.replace(".csv", "_ml_ready.csv")
    df[ml_cols].to_csv(out_ml, index=False)

    print(out_intermediate)
    print(out_ml)

if __name__ == "__main__":
    main()
