#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import pandas as pd
from utils import (
    load_ml_ready,
    build_treated_events,
    build_control_events,
    treated_cluster_support,
)

L = 3
H_MIN = 1
H_MAX = 14
MIN_TREATED_CLUSTERS = 10
MIN_EFFECTIVE_CLUSTERS = 8.0
BETA_CI_WIDTH = 2.0
# MODELS = ["lasso", "ridge", ""elasticnet", "rf", "svr", "xgb", "catboost", "mlp"]
SCRIPT = "t_learner.py"
DATA = "../Data/Symptomtrackingdata_csv-cleaned_with_vars_ml_ready.csv"
OUT_BASE = "../outputs/Results_TLearner_W_t"
ID_COL = "id"
DAY_COL = "day"
OUTCOME_COL = "W_t"
MODELS = "lasso"
TEST_SIZE = "0.2"

def ci_width_from_bootstrap(csv_path):
    bs = pd.read_csv(csv_path)
    for col in ["ATE_bs", "ATE", "ate", "ate_hat", "tau_hat"]:
        if col in bs.columns:
            x = pd.to_numeric(bs[col], errors="coerce").dropna().to_numpy()
            if len(x) == 0:
                return None
            qlo = float(pd.Series(x).quantile(0.025))
            qhi = float(pd.Series(x).quantile(0.975))
            return qhi - qlo
    x = pd.to_numeric(bs.iloc[:, 0], errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return None
    qlo = float(pd.Series(x).quantile(0.025))
    qhi = float(pd.Series(x).quantile(0.975))
    return qhi - qlo

def main():
    df = load_ml_ready(DATA)
    ci_width_ref = None

    for H in range(H_MIN, H_MAX + 1):
        treated = build_treated_events(
            df,
            L=L,
            H=H,
            id_col=ID_COL,
            day_col=DAY_COL,
            treat_col="is_treated",
            treat_type_col="treatment_type",
            outcome_col=OUTCOME_COL,
            inj_codes=(1, 2),
        )
        control = build_control_events(
            df,
            L=L,
            H=H,
            id_col=ID_COL,
            day_col=DAY_COL,
            treat_col="is_treated",
            outcome_col=OUTCOME_COL,
            max_per_id=None,
        )
        events = pd.concat([treated, control], ignore_index=True)
        events = events.dropna(subset=["Y_star"])
        G, G_eff = treated_cluster_support(events, id_col=ID_COL, treat_col="A_star")
        out_dir = Path(OUT_BASE) / f"H{H:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"Running T-learner | L={L}, H={H}")
        print("Output ->", out_dir)
        print(f"[INFO] H={H:02d} | treated patients G={G} | effective G_eff={G_eff:.2f}")
        print("=" * 80)
        cmd = [
            sys.executable,
            SCRIPT,
            "--data", DATA,
            "--out_dir", str(out_dir),
            "--id_col", ID_COL,
            "--day_col", DAY_COL,
            "--L", str(L),
            "--H", str(H),
            "--outcome_col", OUTCOME_COL,
            "--models", MODELS,
            "--test_size", TEST_SIZE,
        ]
        subprocess.run(cmd, check=True)
        bs_path = out_dir / MODELS / "bootstrap_ate_samples.csv"
        ci_w = ci_width_from_bootstrap(bs_path) if bs_path.exists() else None

        if H == H_MIN and ci_w is not None:
            ci_width_ref = ci_w
        low_conf_support = (G < MIN_TREATED_CLUSTERS) or (G_eff < MIN_EFFECTIVE_CLUSTERS)
        low_conf_ci = (ci_width_ref is not None and ci_w is not None and ci_w > BETA_CI_WIDTH * ci_width_ref)
        if ci_width_ref is None:
            if ci_w is None:
                ci_msg = "[CI] missing"
            else:
                ci_msg = f"[CI] width={ci_w:.4f} (no ref)"
        else:
            if ci_w is None:
                ci_msg = f"[CI] missing | ref={ci_width_ref:.4f}"
            else:
                ci_msg = f"[CI] width={ci_w:.4f} | ref(H{H_MIN:02d})={ci_width_ref:.4f} | beta={BETA_CI_WIDTH:.2f}"

        if low_conf_support or low_conf_ci:
            reasons = []
            if low_conf_support:
                reasons.append(f"support(G={G},G_eff={G_eff:.2f})")
            if low_conf_ci:
                reasons.append("CI_width_inflation")
            print(f"[LOW CONFIDENCE] H={H:02d} | " + ", ".join(reasons) + " | " + ci_msg)
        else:
            print(f"[OK] H={H:02d} | " + ci_msg)
    print("\nAll horizons finished successfully.")

if __name__ == "__main__":
    main()
