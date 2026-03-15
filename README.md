# Healthcare Fraud-Risk Tutorial (Run Instructions)

## 1) Prerequisites
- Python 3.10+ recommended
- Required files in this folder:
  - `DIAGNOSES_ICD.csv.gz`
  - `D_ICD_DIAGNOSES.csv.gz`
  - `NOTEEVENTS.csv.gz`

## 2) Install dependencies
```bash
pip install -r requirements_fraud_tutorial.txt
```

## 3) Run the project

Quick run (faster, skips note features):
```bash
python fraud_detector.py --skip-notes --output-dir outputs_quick
```

Full run (recommended for final results):
```bash
python fraud_detector.py --output-dir outputs_full
```

Optional smoke test for large NOTEVENTS processing:
```bash
python fraud_detector.py --max-note-chunks 2 --notes-chunksize 150000 --output-dir outputs_notes_smoke
```

## 4) Key output files
Generated inside your chosen `--output-dir`:
- `model_metrics.json`
- `top_suspicious_admissions.csv`
- `logistic_feature_coefficients.csv`
- `weak_fraud_score_histogram.png`
- `admission_features_with_weak_labels.csv`

