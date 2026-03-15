"""
Self-learning tutorial: Healthcare fraud risk modeling with MIMIC data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DIAGNOSES_FILE = "DIAGNOSES_ICD.csv.gz"
DIAG_DICT_FILE = "D_ICD_DIAGNOSES.csv.gz"
NOTES_FILE = "NOTEEVENTS.csv.gz"


@dataclass
class Paths:
    data_dir: Path
    output_dir: Path
    diagnoses_path: Path
    diagnosis_dictionary_path: Path
    notes_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-learning fraud risk detector using MIMIC admissions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing DIAGNOSES_ICD.csv.gz, D_ICD_DIAGNOSES.csv.gz, NOTEEVENTS.csv.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Folder where model artifacts and tables are saved.",
    )
    parser.add_argument(
        "--skip-notes",
        action="store_true",
        help="Skip NOTEVENTS feature extraction (faster).",
    )
    parser.add_argument(
        "--notes-chunksize",
        type=int,
        default=200_000,
        help="Rows per chunk for NOTEVENTS processing.",
    )
    parser.add_argument(
        "--max-note-chunks",
        type=int,
        default=None,
        help="Optional cap for number of NOTEVENTS chunks (for quick demo runs).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.03,
        help="Target fraction of admissions labeled as suspicious.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="How many highest-risk admissions to export.",
    )
    return parser.parse_args()


def build_paths(data_dir: Path, output_dir: Path) -> Paths:
    return Paths(
        data_dir=data_dir,
        output_dir=output_dir,
        diagnoses_path=data_dir / DIAGNOSES_FILE,
        diagnosis_dictionary_path=data_dir / DIAG_DICT_FILE,
        notes_path=data_dir / NOTES_FILE,
    )


def assert_inputs_exist(paths: Paths, skip_notes: bool) -> None:
    required = [paths.diagnoses_path, paths.diagnosis_dictionary_path]
    if not skip_notes:
        required.append(paths.notes_path)
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_text = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required file(s):\n{missing_text}")


def _clean_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _entropy(values: Iterable[str]) -> float:
    series = pd.Series(list(values))
    probs = series.value_counts(normalize=True)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def build_diagnosis_features(paths: Paths) -> pd.DataFrame:
    print(f"[1/6] Loading diagnosis rows from {paths.diagnoses_path.name} ...")
    diag = pd.read_csv(
        paths.diagnoses_path,
        compression="gzip",
        usecols=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
    )
    diag = diag.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]).copy()
    diag["SUBJECT_ID"] = diag["SUBJECT_ID"].astype(int)
    diag["HADM_ID"] = diag["HADM_ID"].astype(int)
    diag["ICD9_CODE"] = _clean_code(diag["ICD9_CODE"])
    diag["SEQ_NUM"] = pd.to_numeric(diag["SEQ_NUM"], errors="coerce").fillna(0)

    # Rare diagnosis codes are used as one signal of unusual coding behavior.
    code_freq = diag["ICD9_CODE"].value_counts()
    rare_cutoff = max(2, int(np.quantile(code_freq.values, 0.10)))
    rare_codes = set(code_freq[code_freq <= rare_cutoff].index)
    diag["is_rare_code"] = diag["ICD9_CODE"].isin(rare_codes).astype(int)
    
    # Supplementary ICD-9 V and E codes may appear in specialized encounters.
    diag["is_external_or_supplementary"] = (
        diag["ICD9_CODE"].str.startswith(("E", "V"))
    ).astype(int)
    diag["prefix3"] = diag["ICD9_CODE"].str[:3]

    print(f"[2/6] Joining ICD dictionary from {paths.diagnosis_dictionary_path.name} ...")
    diag_dict = pd.read_csv(
        paths.diagnosis_dictionary_path,
        compression="gzip",
        usecols=["ICD9_CODE", "LONG_TITLE"],
    )
    diag_dict["ICD9_CODE"] = _clean_code(diag_dict["ICD9_CODE"])
    diag_dict["LONG_TITLE"] = diag_dict["LONG_TITLE"].fillna("")
    title_lower = diag_dict["LONG_TITLE"].str.lower()
    diag_dict["title_word_count"] = diag_dict["LONG_TITLE"].str.split().str.len().fillna(0)
    diag_dict["is_unspecified"] = title_lower.str.contains(
        r"unspecified|not otherwise specified|\bnos\b", regex=True
    ).astype(int)

    diag = diag.merge(
        diag_dict[["ICD9_CODE", "title_word_count", "is_unspecified"]],
        on="ICD9_CODE",
        how="left",
    )
    diag["title_word_count"] = diag["title_word_count"].fillna(0)
    diag["is_unspecified"] = diag["is_unspecified"].fillna(0)

    # Patient history featuress
    patient_admission_counts = (
        diag[["SUBJECT_ID", "HADM_ID"]]
        .drop_duplicates()
        .groupby("SUBJECT_ID")["HADM_ID"]
        .nunique()
        .rename("patient_admission_count")
    )

    admission_subject = (
        diag.groupby("HADM_ID", as_index=False)["SUBJECT_ID"].first()
        .merge(
            patient_admission_counts.reset_index(),
            on="SUBJECT_ID",
            how="left",
        )
        .set_index("HADM_ID")
    )

    # Core admission-level diagnosis features
    grouped = diag.groupby("HADM_ID")
    diagnosis_features = grouped.agg(
        SUBJECT_ID=("SUBJECT_ID", "first"),
        total_diagnosis_rows=("ICD9_CODE", "size"),
        unique_diagnosis_codes=("ICD9_CODE", "nunique"),
        max_seq_num=("SEQ_NUM", "max"),
        mean_seq_num=("SEQ_NUM", "mean"),
        rare_code_fraction=("is_rare_code", "mean"),
        supplementary_code_fraction=("is_external_or_supplementary", "mean"),
        unspecified_fraction=("is_unspecified", "mean"),
        avg_title_word_count=("title_word_count", "mean"),
    )

    entropy_by_hadm = (
        grouped["prefix3"].apply(_entropy).rename("diagnosis_prefix_entropy")
    )
    diagnosis_features = diagnosis_features.join(entropy_by_hadm, how="left")
    diagnosis_features["duplicate_code_ratio"] = (
        diagnosis_features["total_diagnosis_rows"]
        - diagnosis_features["unique_diagnosis_codes"]
    ) / diagnosis_features["total_diagnosis_rows"].clip(lower=1)

    diagnosis_features = diagnosis_features.join(
        admission_subject[["patient_admission_count"]], how="left"
    )
    diagnosis_features["patient_admission_count"] = diagnosis_features[
        "patient_admission_count"
    ].fillna(1)
    diagnosis_features = diagnosis_features.reset_index()

    return diagnosis_features


def build_note_features(
    notes_path: Path,
    chunksize: int,
    max_chunks: int | None,
) -> pd.DataFrame:
    print(f"[3/6] Streaming note features from {notes_path.name} ...")
    agg_frames: list[pd.DataFrame] = []
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            notes_path,
            compression="gzip",
            usecols=["HADM_ID", "CATEGORY", "TEXT"],
            chunksize=chunksize,
        ),
        start=1,
    ):
        if max_chunks is not None and chunk_idx > max_chunks:
            print(f"      Reached --max-note-chunks={max_chunks}, stopping early.")
            break

        chunk = chunk.dropna(subset=["HADM_ID"]).copy()
        if chunk.empty:
            continue
        chunk["HADM_ID"] = chunk["HADM_ID"].astype(int)
        chunk["CATEGORY"] = chunk["CATEGORY"].fillna("").str.lower()
        chunk["TEXT"] = chunk["TEXT"].fillna("")
        chunk["text_char_len"] = chunk["TEXT"].str.len().astype(float)
        chunk["is_discharge"] = chunk["CATEGORY"].eq("discharge summary").astype(int)

        grouped = chunk.groupby("HADM_ID").agg(
            note_count=("HADM_ID", "size"),
            text_char_sum=("text_char_len", "sum"),
            discharge_note_count=("is_discharge", "sum"),
        )
        agg_frames.append(grouped)

        if chunk_idx % 5 == 0:
            print(f"      Processed {chunk_idx} chunks of notes ...")

    if not agg_frames:
        return pd.DataFrame(columns=["HADM_ID", "note_count", "avg_note_char_len", "discharge_note_fraction"])

    notes = pd.concat(agg_frames).groupby(level=0, as_index=True).sum()
    notes["avg_note_char_len"] = notes["text_char_sum"] / notes["note_count"].clip(lower=1)
    notes["discharge_note_fraction"] = (
        notes["discharge_note_count"] / notes["note_count"].clip(lower=1)
    )
    notes = notes.reset_index()
    return notes[["HADM_ID", "note_count", "avg_note_char_len", "discharge_note_fraction"]]


def generate_weak_labels(
    feature_df: pd.DataFrame,
    contamination: float,
    random_state: int,
) -> pd.DataFrame:
    if not 0.0 < contamination < 0.5:
        raise ValueError("--contamination must be between 0 and 0.5 (exclusive).")

    # Weighted combination of suspicious coding/documentation patterns.
    risk_feature_weights = {
        "unique_diagnosis_codes": 0.20,
        "rare_code_fraction": 0.20,
        "duplicate_code_ratio": 0.15,
        "patient_admission_count": 0.10,
        "supplementary_code_fraction": 0.10,
        "diagnosis_prefix_entropy": 0.10,
        "note_count": 0.075,
        "avg_note_char_len": 0.075,
    }

    available = [col for col in risk_feature_weights if col in feature_df.columns]
    if not available:
        raise ValueError("No risk features available to generate weak labels.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df[available].fillna(0))
    weights = np.array([risk_feature_weights[col] for col in available], dtype=float)
    weights = weights / weights.sum()

    rng = np.random.default_rng(random_state)
    noise = rng.normal(0.0, 0.05, size=len(feature_df))
    risk_score = scaled @ weights + noise

    threshold = float(np.quantile(risk_score, 1.0 - contamination))
    weak_label = (risk_score >= threshold).astype(int)

    labeled = feature_df.copy()
    labeled["weak_fraud_score"] = risk_score
    labeled["weak_fraud_label"] = weak_label
    labeled["weak_label_threshold"] = threshold
    return labeled


def train_and_evaluate_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float,
    contamination: float,
    random_state: int,
) -> tuple[dict, Pipeline, IsolationForest, pd.DataFrame]:
    X = df[feature_cols].fillna(0).copy()
    y = df["weak_fraud_label"].astype(int).copy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Supervised baseline model (uses weak labels).
    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    logistic.fit(X_train, y_train)
    lr_prob = logistic.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)

    # Unsupervised anomaly detector (does not use labels).
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    iso.fit(X_train)
    # Higher score should mean "more suspicious", so negate decision_function.
    iso_score = -iso.decision_function(X_test)
    iso_pred = (iso.predict(X_test) == -1).astype(int)

    metrics = {
        "dataset_summary": {
            "admissions": int(len(df)),
            "feature_count": int(len(feature_cols)),
            "weak_positive_rate": float(y.mean()),
        },
        "logistic_regression": {
            "roc_auc": float(roc_auc_score(y_test, lr_prob)),
            "average_precision": float(average_precision_score(y_test, lr_prob)),
            "confusion_matrix": confusion_matrix(y_test, lr_pred).tolist(),
            "classification_report": classification_report(
                y_test, lr_pred, output_dict=True, zero_division=0
            ),
        },
        "isolation_forest": {
            "roc_auc": float(roc_auc_score(y_test, iso_score)),
            "average_precision": float(average_precision_score(y_test, iso_score)),
            "confusion_matrix": confusion_matrix(y_test, iso_pred).tolist(),
            "classification_report": classification_report(
                y_test, iso_pred, output_dict=True, zero_division=0
            ),
        },
    }

    test_predictions = df.loc[idx_test, ["HADM_ID", "SUBJECT_ID", "weak_fraud_score", "weak_fraud_label"]].copy()
    test_predictions["logistic_prob"] = lr_prob
    test_predictions["isolation_score"] = iso_score
    test_predictions["logistic_pred"] = lr_pred
    test_predictions["isolation_pred"] = iso_pred
    test_predictions["combined_rank_score"] = (
        test_predictions["logistic_prob"].rank(pct=True)
        + test_predictions["isolation_score"].rank(pct=True)
    ) / 2.0
    test_predictions = test_predictions.sort_values("combined_rank_score", ascending=False)

    return metrics, logistic, iso, test_predictions


def export_diagnostics(
    labeled_df: pd.DataFrame,
    feature_cols: list[str],
    metrics: dict,
    logistic: Pipeline,
    iso: IsolationForest,
    ranked_predictions: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    labeled_df.to_csv(output_dir / "admission_features_with_weak_labels.csv", index=False)
    ranked_predictions.head(top_n).to_csv(
        output_dir / "top_suspicious_admissions.csv", index=False
    )

    # Logistic coefficient interpretation (standardized features).
    model = logistic.named_steps["model"]
    coefs = pd.DataFrame(
        {"feature": feature_cols, "coefficient": model.coef_.ravel()}
    ).sort_values("coefficient", ascending=False)
    coefs.to_csv(output_dir / "logistic_feature_coefficients.csv", index=False)

    joblib.dump(logistic, output_dir / "logistic_regression.joblib")
    joblib.dump(iso, output_dir / "isolation_forest.joblib")

    # Plot weak-label risk score distribution for slide visuals.
    plt.figure(figsize=(8, 4.5))
    labeled_df["weak_fraud_score"].hist(bins=60)
    plt.title("Distribution of Weak Fraud Scores (Admission-Level)")
    plt.xlabel("Weak Fraud Score")
    plt.ylabel("Number of Admissions")
    plt.tight_layout()
    plt.savefig(output_dir / "weak_fraud_score_histogram.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    paths = build_paths(args.data_dir, args.output_dir)
    assert_inputs_exist(paths, skip_notes=args.skip_notes)

    diagnosis_features = build_diagnosis_features(paths)

    if args.skip_notes:
        print("[3/6] Skipping note features (--skip-notes enabled).")
        full_features = diagnosis_features.copy()
    else:
        note_features = build_note_features(
            paths.notes_path,
            chunksize=args.notes_chunksize,
            max_chunks=args.max_note_chunks,
        )
        full_features = diagnosis_features.merge(note_features, on="HADM_ID", how="left")

    # Fill missing note features when notes are absent for an admission.
    for col in ["note_count", "avg_note_char_len", "discharge_note_fraction"]:
        if col in full_features.columns:
            full_features[col] = full_features[col].fillna(0)

    print("[4/6] Creating weak fraud labels for tutorial supervision ...")
    labeled = generate_weak_labels(
        full_features,
        contamination=args.contamination,
        random_state=args.random_state,
    )

    exclude_cols = {"HADM_ID", "SUBJECT_ID", "weak_fraud_label", "weak_fraud_score", "weak_label_threshold"}
    feature_cols = [c for c in labeled.columns if c not in exclude_cols]

    print("[5/6] Training logistic regression + isolation forest ...")
    metrics, logistic, iso, ranked_predictions = train_and_evaluate_models(
        labeled,
        feature_cols=feature_cols,
        test_size=args.test_size,
        contamination=args.contamination,
        random_state=args.random_state,
    )

    print(f"[6/6] Saving outputs to {paths.output_dir} ...")
    export_diagnostics(
        labeled_df=labeled,
        feature_cols=feature_cols,
        metrics=metrics,
        logistic=logistic,
        iso=iso,
        ranked_predictions=ranked_predictions,
        output_dir=paths.output_dir,
        top_n=args.top_n,
    )

    print("Done.")
    print(f"Admissions modeled: {metrics['dataset_summary']['admissions']:,}")
    print(f"Weak fraud positive rate: {metrics['dataset_summary']['weak_positive_rate']:.4f}")
    print(
        "Logistic ROC-AUC: "
        f"{metrics['logistic_regression']['roc_auc']:.4f} | "
        "IsolationForest ROC-AUC: "
        f"{metrics['isolation_forest']['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
