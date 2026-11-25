# Model Card — Machine Fault Intelligence

## Summary

This project contains supervised failure classifiers and an unsupervised anomaly detector trained from the AI4I 2020 Predictive Maintenance Dataset. The dashboard's primary failure-risk model is HistGradientBoosting in a scaler/SMOTE pipeline. The Phase 1 Random Forest is a retained comparison. Isolation Forest provides a separate unusual-condition signal.

## Intended use

- Academic demonstration of imbalanced classification, leakage-aware validation, PCA comparison, anomaly detection, and a local inference interface.
- Exploratory assessment of records shaped like the five AI4I sensor inputs.
- Human-reviewed triage demonstration where an anomaly is understood as unusual behavior rather than proof of failure.

## Out-of-scope use

- Production industrial deployment or safety-critical decisions.
- Real-time or connected machine monitoring.
- Remaining useful life (RUL), time-to-failure, or maintenance scheduling prediction.
- Diagnosis of a specific failure mode, causal inference, or replacement of engineering inspection.
- Claims of calibrated real-world failure probabilities.

## Data characteristics

The source is the [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i), DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C). It has 10,000 synthetic observations and 339 recorded failures in the checked project copy (3.39%). UCI explicitly describes the dataset as synthetic. Its distributions, label-generation process, and machine coverage may not represent a specific plant.

## Inputs and target

Five numeric inputs in fixed order:

1. Air temperature [K]
2. Process temperature [K]
3. Rotational speed [rpm]
4. Torque [Nm]
5. Tool wear [min]

Supervised target: `Machine failure` (0 = no recorded failure, 1 = recorded failure). UDI, Product ID, Type, target-derived failure mode fields (TWF, HDF, PWF, OSF, RNF) are not model inputs.

## Models

- **Primary HGB:** StandardScaler → SMOTE(random_state=42) → HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=42). The application uses its standard `predict()` result for the FAILURE RISK state. `predict_proba` is displayed as an uncalibrated model score.
- **Random Forest comparison:** StandardScaler → SMOTE(random_state=42) → RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).
- **Isolation Forest:** StandardScaler fitted on normal development rows → IsolationForest(n_estimators=300, contamination=0.05, random_state=42, n_jobs=-1). The score is `-score_samples`; the saved cutoff is 0.571183.
- **Retained experiment:** Random Forest on original five sensor features versus a pairwise PCA representation.

## Evaluation methodology

The notebook and artifact script use the stratified 75/25 split with seed 42: 7,500 development rows and 2,500 holdout rows (85 holdout failures). Scaling and SMOTE are fitted only on development data, and SMOTE is inside fold-local imbalanced-learn pipelines in CV. RandomizedSearchCV and threshold/contamination calibration use development-only data. The artifact script fits the HGB and RF pipelines on the 7,500 training rows. The IF model fits only normal rows from a training-only sub-split. The holdout is for final metrics and descriptive evaluation only.

Metrics include precision, recall, F1, PR-AUC/average precision, confusion matrix, FPR and FNR. The IF “failure” metrics use `Machine failure` as a proxy label, because no independent anomaly ground truth is provided. Hard OR alerts have no ranking score; AUC is not reported for them.

## Saved holdout performance

| Model | Precision | Recall | F1 | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| HGB primary | 0.5726 | 0.8353 | 0.6794 | 0.8039 | 0.0219 | 0.1647 |
| Random Forest baseline | 0.5231 | 0.8000 | 0.6326 | 0.7103 | 0.0257 | 0.2000 |
| Isolation Forest calibrated* | 0.1288 | 0.2000 | 0.1567 | 0.1385 | 0.0476 | 0.8000 |
| HGB OR Isolation Forest* | 0.3133 | 0.8588 | 0.4591 | — | 0.0663 | 0.1412 |

*For IF and combined policy, “failure” metrics compare alerts with the recorded failure target; anomaly truth is unavailable. Combined OR PR-AUC is undefined because it outputs a binary alert rather than a continuous ranking score.*

## Known limitations and risks

- AI4I data is synthetic; no real equipment validation has been performed.
- A random split may overstate performance for temporal or new-machine deployment.
- Scores are not separately calibrated; decision thresholds do not encode explicit maintenance costs.
- SMOTE may create implausible combinations in sensor space.
- Failure modes and operational context are not explicitly modeled.
- IF anomaly score is not a failure probability; its proxy metrics do not measure anomaly truth.
- Holdout permutation importance is descriptive post-hoc reporting only and must not be used to select features or models.
- Explanations are global feature importance and sensor percentile context; neither proves causality.
- No RUL, fault diagnosis, monitoring, drift handling, authentication, or production safety validation exists.

## Responsible interpretation

Use FAILURE RISK as a model alert for review, ANOMALOUS CONDITION as an unusual-sensor-pattern prompt, and NORMAL CONDITION as lack of either model alert—not as proof that a machine is healthy. A missed failure and an unnecessary alert both matter; operating thresholds should only be selected with a representative development calibration process and explicit maintenance costs. Do not present this prototype as validated for production industrial deployment.
