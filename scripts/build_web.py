import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ai4i+2020+predictive+maintenance+dataset" / "ai4i2020.csv"
MODELS = ROOT / "models"
WEB = ROOT / "web"


def floats(values):
    return [float(v) for v in values]


def rounded(values, digits):
    return [round(float(v), digits) for v in values]


def hgb_trees(model):
    trees = []
    for iteration in model._predictors:
        nodes = iteration[0].nodes
        trees.append({
            "f": [-1 if leaf else int(f) for f, leaf in zip(nodes["feature_idx"], nodes["is_leaf"])],
            "t": floats(nodes["num_threshold"]),
            "l": [int(v) for v in nodes["left"]],
            "r": [int(v) for v in nodes["right"]],
            "v": rounded(nodes["value"], 7),
        })
    return trees


def avg_path(n):
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    out[n == 2] = 1
    big = n > 2
    out[big] = 2 * (np.log(n[big] - 1) + np.euler_gamma) - 2 * (n[big] - 1) / n[big]
    return out


def iso_trees(model):
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        leaf = tree.children_left == -1
        trees.append({
            "f": [-1 if is_leaf else int(f) for f, is_leaf in zip(tree.feature, leaf)],
            "t": floats(tree.threshold),
            "l": [int(v) for v in tree.children_left],
            "r": [int(v) for v in tree.children_right],
            "a": rounded(np.where(leaf, avg_path(tree.n_node_samples), 0), 4),
        })
    return trees


def main():
    meta = json.loads((MODELS / "metadata.json").read_text())
    features = meta["feature_order"]
    data = pd.read_csv(DATA).drop_duplicates().dropna(subset=features + ["Machine failure"])
    X = data[features].astype(float)
    y = data["Machine failure"].astype(int)
    _, X_hold, _, y_hold = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    hgb_pipe = joblib.load(MODELS / "hist_gradient_boosting_pipeline.joblib")
    iso_pipe = joblib.load(MODELS / "isolation_forest_pipeline.joblib")
    hgb_scaler, hgb = hgb_pipe.named_steps["scale"], hgb_pipe.named_steps["model"]
    iso_scaler, iso = iso_pipe.named_steps["scale"], iso_pipe.named_steps["model"]
    cutoff = meta["thresholds"]["isolation_forest_anomaly_score"]

    bundle = {
        "features": features,
        "hgb": {
            "mean": floats(hgb_scaler.mean_),
            "scale": floats(hgb_scaler.scale_),
            "base": float(hgb._baseline_prediction.ravel()[0]),
            "trees": hgb_trees(hgb),
        },
        "iso": {
            "mean": floats(iso_scaler.mean_),
            "scale": floats(iso_scaler.scale_),
            "cm": float(avg_path([iso.max_samples_])[0]),
            "cut": cutoff,
            "trees": iso_trees(iso),
        },
        "range": {f: [float(data[f].min()), float(data[f].max())] for f in features},
        "median": {f: float(data[f].median()) for f in features},
    }

    hgb_score = hgb.predict_proba(hgb_scaler.transform(X_hold))[:, 1]
    iso_score = -iso_pipe.score_samples(X_hold)
    labels = y_hold.values

    def pick(mask, name, desc):
        idx = np.where(mask)[0]
        return {"name": name, "desc": desc, "v": floats(X_hold.values[idx[len(idx) // 3]])}

    bundle["presets"] = [
        pick((hgb_score < 0.5) & (iso_score < cutoff) & (labels == 0), "Routine run", "Healthy row from the holdout set"),
        pick((hgb_score > 0.9) & (labels == 1), "Recorded failure", "Failure row the model catches"),
        pick((hgb_score < 0.5) & (iso_score > cutoff + 0.02) & (labels == 0), "Unusual but fine", "No failure recorded, odd readings"),
        pick((hgb_score < 0.5) & (labels == 1), "Missed failure", "Failure row the model misses"),
    ]

    template = (WEB / "template.html").read_text(encoding="utf-8")
    page = template.replace("__MODEL__", json.dumps(bundle, separators=(",", ":")))
    (WEB / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {WEB / 'index.html'} ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
