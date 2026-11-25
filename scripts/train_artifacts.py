from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.inference import DEFAULT_DATA_PATH, FEATURES, TARGET  # noqa: E402

RANDOM_STATE = 42
HGB_THRESHOLD = 0.5
IF_CONTAMINATION = 0.05
IF_SCORE_THRESHOLD = 0.571183


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    if not DEFAULT_DATA_PATH.is_file():
        raise FileNotFoundError(f"Expected the dataset at {DEFAULT_DATA_PATH}")
    data = pd.read_csv(DEFAULT_DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    clean = data.drop_duplicates().dropna(subset=required).copy()
    X = clean[FEATURES].astype(float)
    y = clean[TARGET].astype(int)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    random_forest = ImbPipeline(steps=[
        ("scale", StandardScaler()),
        ("sampler", SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto")),
        ("model", RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    random_forest.fit(X_train, y_train)

    hist_gradient_boosting = ImbPipeline(steps=[
        ("scale", StandardScaler()),
        ("sampler", SMOTE(random_state=RANDOM_STATE)),
        ("model", HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, random_state=RANDOM_STATE
        )),
    ])
    hist_gradient_boosting.fit(X_train, y_train)

    X_if_fit, X_if_calibration, y_if_fit, y_if_calibration = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train
    )
    normal_development = X_if_fit.loc[y_if_fit == 0]
    isolation_forest = Pipeline(steps=[
        ("scale", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=300, contamination=IF_CONTAMINATION,
            random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    isolation_forest.fit(normal_development)

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "hist_gradient_boosting_pipeline.joblib": hist_gradient_boosting,
        "random_forest_pipeline.joblib": random_forest,
        "isolation_forest_pipeline.joblib": isolation_forest,
    }
    for filename, model in artifacts.items():
        joblib.dump(model, models_dir / filename, compress=3)

    normal_reference = {}
    for feature in FEATURES:
        normal_reference[feature] = {
            "p05": float(normal_development[feature].quantile(0.05)),
            "median": float(normal_development[feature].median()),
            "p95": float(normal_development[feature].quantile(0.95)),
        }
    versions = {
        package: package_version(package)
        for package in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "joblib"]
    }
    versions["python"] = platform.python_version()
    metadata = {
        "schema_version": 1,
        "project": "Machine Fault Intelligence",
        "feature_order": FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "thresholds": {
            "hist_gradient_boosting_failure": HGB_THRESHOLD,
            "isolation_forest_contamination": IF_CONTAMINATION,
            "isolation_forest_anomaly_score": IF_SCORE_THRESHOLD,
            "isolation_forest_score_definition": "-score_samples; higher means more unusual",
        },
        "training": {
            "dataset_path": str(DEFAULT_DATA_PATH.relative_to(ROOT)),
            "dataset_sha256": hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest(),
            "clean_observations": int(len(clean)),
            "development_rows": int(len(X_train)),
            "holdout_rows_excluded_from_fit": int(len(X_holdout)),
            "if_normal_fit_rows": int(len(normal_development)),
            "if_calibration_rows_not_used_for_fit": int(len(X_if_calibration)),
            "fit_scope": "HistGradientBoosting and Random Forest use X_train only; Isolation Forest uses normal rows from its X_train development slice only.",
        },
        "models": {
            "hist_gradient_boosting": {
                "pipeline": ["StandardScaler", "SMOTE(random_state=42)", "HistGradientBoostingClassifier"],
                "max_iter": 150,
                "learning_rate": 0.1,
                "random_state": RANDOM_STATE,
                "decision_rule": "predict() default class decision (0.5 binary threshold)",
            },
            "random_forest": {
                "pipeline": ["StandardScaler", "SMOTE(random_state=42)", "RandomForestClassifier"],
                "n_estimators": 300,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
            "isolation_forest": {
                "pipeline": ["StandardScaler fit on normal development rows", "IsolationForest"],
                "n_estimators": 300,
                "contamination": IF_CONTAMINATION,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "fit_scope": "normal development rows only",
            },
        },
        "normal_sensor_reference": normal_reference,
        "package_versions": versions,
        "evaluation_file": "evaluation_results.json",
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Trained and saved {len(artifacts)} artifacts in {models_dir}")
    print(f"Training rows: {len(X_train)}; excluded holdout rows: {len(X_holdout)}")
    print(f"Isolation Forest fit rows (normal only): {len(normal_development)}")


if __name__ == "__main__":
    main()
