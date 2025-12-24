# Machine learning for dynamic symptom prediction and treatment effect estimation in pernicious anaemia

---

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Domain](https://img.shields.io/badge/domain-clinical%20ML%20%7C%20causal%20inference-blue.svg)]()
[![Methods](https://img.shields.io/badge/methods-forecasting%20%7C%20DR--ATE-informational.svg)]()
[![Language](https://img.shields.io/badge/language-Python-3776AB.svg)]()
[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-3776AB.svg)]()
[![ML Stack](https://img.shields.io/badge/ML%20stack-scikit--learn%20%7C%20PyTorch-FF6F00.svg)]()
[![Reproducibility](https://img.shields.io/badge/reproducibility-containerised%20%7C%20deterministic-success.svg)]()
[![Containers](https://img.shields.io/badge/containers-Docker%20%7C%20Conda-2496ED.svg)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20WSL2-lightgrey.svg)]()
[![Docs](https://img.shields.io/badge/docs-auto--generated-green.svg)]()

> *Time-series + Causal Machine Learning*

---

**Authors:**  
> Guillermo Comesaña Cimadevila · Alfie Thain · Kourosh Ahmadi

---

Thain, A., Comesaña Cimadevila, G., *et al.* **Symptom-Tracking Pipeline**: Machine learning reveals symptom dynamics and treatment effects in pernicious anaemia. *GitHub repository*. https://github.com/guillermocomesanacimadevila/PA-Symptom-Tracking.

---

```bash
git clone https://github.com/guillermocomesanacimadevila/PA-Symptom-Tracking.git
```

```bash
cd PA-Symptom-Tracking
```

```bash
conda env create -f env/environment.yml && conda activate symptom-tracking
```

```bash
cd Scripts/ && chmod +x run_all_models.py && python run_all_models.py 
```

# Changing directories

```bash
cd PA-Symptom-Tracking/Data && pwd && ls -lh
```

```bash
cd ../Scripts && chmod +x run_all_models.py && python run_all_models.py 
```

# Test run

```bash
git clone https://github.com/guillermocomesanacimadevila/PA-Symptom-Tracking.git
```

```bash
cd PA-Symptom-Tracking
```

```bash
conda env create -f env/environment.yml && conda activate symptom-tracking
```

```bash
cd Scripts/ && chmod +x test.py && python test.py 
```

# Data prep & Hmax

```bash
python data_prep.py --input ../Data/Symptomtrackingdata_csv-cleaned_with_vars.csv --h-max hmax
```

```bash
python run_all_models.py \
  --data ../Data/Symptomtrackingdata_csv-cleaned_with_vars_ml_ready.csv \
  --out_dir ../grid_results_horizons \
  --target W \
  --summarise-by patient \
  --h-max auto \
  --h-cap 9999
```

# Bootsraping


```bash
python bootstrap.py \
  --pred_dir ../grid_results_horizons \
  --models auto \
  --h_start 1 \
  --h_end 14 \
  --n_boot 2000 \
  --out_csv ../grid_results_horizons/ci_all_models.csv
```

# Predict next-day W_t or S_t

```bash
python run_all_models.py \
  --data ../Data/Symptomtrackingdata_csv-cleaned_with_vars_ml_ready.csv \
  --out_dir ../Data/grid_results_Ut \
  --use_delta \
  --target U_t
```


## Run Causal ML!

```bash
python t_learner.py \
  --data ../Data/Symptomtrackingdata_csv-cleaned_with_vars_ml_ready.csv \
  --out_dir ../outputs/Results_TLearner \
  --id_col id \
  --day_col day \
  --L 3 \
  --H 3 \
  --outcome_col W_t \
  --models lasso \
  --test_size 0.2
```

```bash
python run_horizons.py
```

## Expl of causal results

```bash
python analysis_complete.py \
  --dir Results_TLearner/lasso \
  --id_col id \
  --bootstrap_reps 1000
```
