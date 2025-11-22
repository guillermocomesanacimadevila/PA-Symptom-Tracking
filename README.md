# PA-Symptom-Tracking
alfie 
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
  --out_dir ../Data/grid_results_S \
  --use_delta \
  --target S_t
```


