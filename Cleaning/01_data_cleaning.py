import re
import numpy as np
import pandas as pd
from typing import List

IN_PATH  = "/Users/c24102394/Downloads/Symptomtrackingdata.csv"
OUT_PATH = "Symptomtrackingdata_cleaned.csv"
SYMPTOM_COLS_CANON = [
    "Tiredness","Poor_concentration","Irritability","Intestinal_problems","Memory_loss",
    "Muscle_pain","Nerve_pain","Pins_and_needles","Tinnitus",
    "Word_finding_difficulties","Dizziness","Overall_wellbeing"
]

RENAME_MAP = {
    "Start Date":"StartDate",
    "Poor concentration":"Poor_concentration",
    "Intestinal problems":"Intestinal_problems",
    "Memory loss":"Memory_loss",
    "Muscle pain":"Muscle_pain",
    "Nerve pain":"Nerve_pain",
    "Pins and needles":"Pins_and_needles",
    "Word finding difficulties":"Word_finding_difficulties",
    "Overall wellbeing":"Overall_wellbeing",
}

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        keep_default_na=False,
        na_values=["", "NA", "NaN", "None"]
    )
    df = df.loc[:, ~df.columns.str.match(r"Unnamed|^t$|^u$|^v$|^w$|^x$|^y$|^z$|^aa$")]
    return df

def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME_MAP)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

def parse_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    if "StartDate" in df.columns:
        df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce", dayfirst=True)

    for c in ["Day","Finished","ID","Treatment","TreatmentType"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "TreatmentType2" in df.columns:
        if pd.api.types.is_numeric_dtype(df["TreatmentType2"]):
            df["TreatmentType2"] = df["TreatmentType2"].apply(lambda x: np.nan if pd.isna(x) else str(int(x)))
        else:
            df["TreatmentType2"] = df["TreatmentType2"].astype(str).str.strip()
            df.loc[df["TreatmentType2"].isin(["", "nan", "None"]), "TreatmentType2"] = np.nan

    df = df.sort_values(["ID","Day"], kind="mergesort").reset_index(drop=True)
    return df

def symptom_columns_present(df: pd.DataFrame) -> List[str]:
    return [c for c in SYMPTOM_COLS_CANON if c in df.columns]

def apply_finished_rule(df: pd.DataFrame, symptoms: List[str]) -> pd.DataFrame:
    for c in symptoms:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Finished" in df.columns:
        df.loc[df["Finished"] == 0, symptoms] = np.nan
    for c in symptoms:
        df[c] = df[c].clip(lower=0, upper=10)
    return df

def had_im_injection_row(row: pd.Series) -> int:
    t   = row.get("Treatment")
    tt1 = row.get("TreatmentType")
    tt2 = row.get("TreatmentType2")
    if t != 1:
        return 0
    if tt1 == 1:
        return 1
    if isinstance(tt2, str) and re.search(r"1", tt2):
        return 1
    return 0

def add_injection_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["Injection"] = df.apply(had_im_injection_row, axis=1).astype(int)
    return df

def compute_days_since_injection(df: pd.DataFrame) -> pd.DataFrame:
    def per_id(g: pd.DataFrame) -> pd.Series:
        inj = g["Injection"].astype(bool)
        epi = inj.cumsum()
        dsi = g.groupby(epi).cumcount().astype(float)
        dsi[(epi == 0) & (~inj)] = np.nan
        dsi.index = g.index
        return dsi

    df["DaysSinceInjection"] = (
        df.groupby("ID", group_keys=False)[["Injection"]]
          .apply(lambda sub: per_id(df.loc[sub.index]))
    )
    return df

def cast_nullable_ints(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Day","ID","Finished","Treatment","TreatmentType"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def deduplicate_by_id_day(df: pd.DataFrame) -> pd.DataFrame:
    if "StartDate" in df.columns:
        df = df.sort_values(["ID","Day","StartDate"])
    df = df.drop_duplicates(subset=["ID","Day"], keep="first")
    return df

def add_quality_flags(df: pd.DataFrame, symptoms: List[str]) -> pd.DataFrame:
    df["any_missing_symptoms"]  = df[symptoms].isna().any(axis=1)
    df["has_any_symptom_value"] = (~df[symptoms].isna()).any(axis=1)
    return df

def add_episode_helpers(df: pd.DataFrame) -> pd.DataFrame:
    df["EpisodeID"] = df.groupby("ID")["Injection"].cumsum()
    if "Day" in df.columns:
        next_inj_day = (
            df.assign(inj_day = df["Day"].where(df["Injection"].eq(1)))
              .groupby("ID")["inj_day"].bfill()
        )
        df["DaysToNextInjection"] = next_inj_day - df["Day"]
    return df

def save_clean(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    df = load_raw(IN_PATH)
    df = tidy_columns(df)
    df = parse_and_cast(df)

    symptoms = symptom_columns_present(df)
    df = apply_finished_rule(df, symptoms)
    df = add_injection_flag(df)
    df = compute_days_since_injection(df)

    df = deduplicate_by_id_day(df)
    df = cast_nullable_ints(df)
    df = add_quality_flags(df, symptoms)
    df = add_episode_helpers(df)

    save_clean(df, OUT_PATH)

    print(f"Saved {OUT_PATH} | rows: {len(df)} | cols: {len(df.columns)}")
    print(f"Participants: {df['ID'].nunique()} | Injection days: {int(df['Injection'].sum())}")
    print("Columns:", list(df.columns))
    print(df.head(5).to_string(index=False))