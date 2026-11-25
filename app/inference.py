from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
RESERVED_OUTPUT_COLUMNS = {
    "HGB failure probability",
    "HGB failure prediction",
    "Random Forest failure probability",
    "Random Forest failure prediction",
    "Isolation Forest anomaly score",
    "Isolation Forest alert",
    "Machine condition",
    "Input validation error",
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_DATA_PATH = ROOT / "ai4i+2020+predictive+maintenance+dataset" / "ai4i2020.csv"


class ArtifactLoadError(RuntimeError):
    """Raised when a required, locally trained model artifact is unavailable."""


def load_artifacts(models_dir: Path | str = DEFAULT_MODELS_DIR) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the trusted model bundle and metadata from the local models directory."""
    models_dir = Path(models_dir)
    metadata_path = models_dir / "metadata.json"
    artifact_paths = {
        "hgb": models_dir / "hist_gradient_boosting_pipeline.joblib",
        "random_forest": models_dir / "random_forest_pipeline.joblib",
        "isolation_forest": models_dir / "isolation_forest_pipeline.joblib",
    }
    missing = [str(path) for path in [metadata_path, *artifact_paths.values()] if not path.is_file()]
    if missing:
        raise ArtifactLoadError(
            "Trained model files are missing. From the project root, run "
            "`python scripts/train_artifacts.py`. Missing: " + ", ".join(missing)
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("feature_order") != FEATURES:
            raise ArtifactLoadError("Artifact feature order does not match the application input schema.")
        models = {name: joblib.load(path) for name, path in artifact_paths.items()}
    except ArtifactLoadError:
        raise
    except Exception as exc:
        raise ArtifactLoadError(
            "Could not load the local model artifacts. Regenerate them with "
            "`python scripts/train_artifacts.py`. Details: " + str(exc)
        ) from exc
    return models, metadata


def load_dataset(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the project's historical AI4I dataset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def validate_sensor_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric, ordered sensor inputs and a per-row validation message."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Sensor input must be a pandas DataFrame.")
    if frame.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names. Rename columns so each is unique.")
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError("Missing required sensor column(s): " + ", ".join(missing))
    reserved = sorted(RESERVED_OUTPUT_COLUMNS.intersection(frame.columns))
    if reserved:
        raise ValueError("Input uses reserved prediction output column name(s): " + ", ".join(reserved))

    numeric = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")
    messages = pd.Series("", index=frame.index, dtype="object")
    for feature in FEATURES:
        values = numeric[feature].to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        messages.loc[bad] = messages.loc[bad].map(
            lambda previous: (previous + "; " if previous else "") + f"{feature} must be a finite number"
        )
        if feature in ("Air temperature [K]", "Process temperature [K]"):
            invalid = np.isfinite(values) & (values <= 0)
            messages.loc[invalid] = messages.loc[invalid].map(
                lambda previous: (previous + "; " if previous else "") + f"{feature} must be greater than 0 K"
            )
        elif feature == "Rotational speed [rpm]":
            invalid = np.isfinite(values) & (values <= 0)
            messages.loc[invalid] = messages.loc[invalid].map(
                lambda previous: (previous + "; " if previous else "") + "Rotational speed must be greater than 0 rpm"
            )
        else:
            invalid = np.isfinite(values) & (values < 0)
            messages.loc[invalid] = messages.loc[invalid].map(
                lambda previous: (previous + "; " if previous else "") + f"{feature} cannot be negative"
            )
    messages = messages.replace("", pd.NA)
    return numeric, messages


def condition_state(failure_prediction: int | bool, anomaly_alert: int | bool) -> str:
    """Apply the project's existing three-state condition rules."""
    if bool(failure_prediction):
        return "FAILURE RISK"
    if bool(anomaly_alert):
        return "ANOMALOUS CONDITION"
    return "NORMAL CONDITION"


def predict_batch(
    original: pd.DataFrame,
    models: dict[str, Any],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Preserve source rows and append predictions; invalid rows remain in the result."""
    numeric, errors = validate_sensor_frame(original)
    result = original.copy()
    for column in RESERVED_OUTPUT_COLUMNS:
        result[column] = pd.NA
    result["Input validation error"] = errors
    valid = errors.isna()
    if not valid.any():
        return result

    sensor_values = numeric.loc[valid, FEATURES]
    hgb = models["hgb"]
    rf = models["random_forest"]
    iso = models["isolation_forest"]
    hgb_classes = list(hgb.classes_)
    rf_classes = list(rf.classes_)
    hgb_failure_probability = hgb.predict_proba(sensor_values)[:, hgb_classes.index(1)]
    rf_failure_probability = rf.predict_proba(sensor_values)[:, rf_classes.index(1)]
    hgb_prediction = hgb.predict(sensor_values).astype(int)
    rf_prediction = rf.predict(sensor_values).astype(int)

    iso_scaler = iso.named_steps["scale"]
    iso_model = iso.named_steps["model"]
    anomaly_score = -iso_model.score_samples(iso_scaler.transform(sensor_values))
    threshold = float(metadata["thresholds"]["isolation_forest_anomaly_score"])
    anomaly_alert = (anomaly_score >= threshold).astype(int)
    statuses = [condition_state(failure, anomaly) for failure, anomaly in zip(hgb_prediction, anomaly_alert)]

    result.loc[valid, "HGB failure probability"] = hgb_failure_probability
    result.loc[valid, "HGB failure prediction"] = hgb_prediction
    result.loc[valid, "Random Forest failure probability"] = rf_failure_probability
    result.loc[valid, "Random Forest failure prediction"] = rf_prediction
    result.loc[valid, "Isolation Forest anomaly score"] = anomaly_score
    result.loc[valid, "Isolation Forest alert"] = anomaly_alert
    result.loc[valid, "Machine condition"] = statuses
    return result


def anomaly_context(sensor_row: pd.Series, metadata: dict[str, Any]) -> list[str]:
    """Describe marginal sensor values outside normal training percentiles."""
    reference = metadata.get("normal_sensor_reference", {})
    notes = []
    for feature in FEATURES:
        bounds = reference.get(feature, {})
        low, high = bounds.get("p05"), bounds.get("p95")
        value = float(sensor_row[feature])
        if low is not None and value < low:
            notes.append(f"{feature} is below the normal-training 5th percentile ({low:.2f}).")
        elif high is not None and value > high:
            notes.append(f"{feature} is above the normal-training 95th percentile ({high:.2f}).")
    if not notes:
        notes.append(
            "No single sensor is outside its normal-training 5th–95th percentile range; "
            "the anomaly score can still reflect a multivariate combination."
        )
    return notes
