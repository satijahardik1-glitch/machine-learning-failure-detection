# Machine Fault Intelligence — Technical Report

**Project snapshot:** Phase 5, 25 November 2026  
**Scope:** Academic prototype using the saved AI4I 2020 dataset and a fixed stratified holdout.

## 1. Problem statement and objectives

The project evaluates whether five machine operating measurements can help classify the AI4I dataset's recorded `Machine failure` target, and whether a separate unsupervised anomaly detector can provide a complementary unusual-condition signal. The accompanying Streamlit application supports offline single-record and batch assessment. It is an educational prototype; it does not monitor connected equipment or estimate remaining useful life (RUL).

## 2. Dataset and source

The project uses the 10,000-row AI4I 2020 Predictive Maintenance Dataset distributed by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i), DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C). UCI describes it as synthetic data reflecting predictive-maintenance patterns. Its overall failure target is rare: the local CSV contains 339 failure rows and 9,661 non-failure rows (3.39% prevalence). The local file has 14 columns and no missing values in the checked source copy.

Dataset citation: UCI Machine Learning Repository. (2020). *AI4I 2020 Predictive Maintenance Dataset* [Dataset]. https://doi.org/10.24432/C5HS5C. Introductory paper listed by UCI: S. Matzka, “Explainable Artificial Intelligence for Predictive Maintenance Applications,” 2020 IEEE International Conference on Artificial Intelligence for Industries, DOI [10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023).

The five deployed predictors are:

| Input | Meaning / unit |
|---|---|
| `Air temperature [K]` | Air temperature, kelvin |
| `Process temperature [K]` | Process temperature, kelvin |
| `Rotational speed [rpm]` | Rotational speed, revolutions per minute |
| `Torque [Nm]` | Torque, newton metres |
| `Tool wear [min]` | Tool wear, minutes |

The target is `Machine failure` (0/1). `UDI`, `Product ID`, `Type`, target, and failure-mode indicators `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are not primary model predictors. `Type` appears in descriptive EDA only. Identifiers and failure-mode flags are excluded to avoid memorization or target-derived leakage.

## 3. Preparation and exploratory analysis

The notebook checks required columns, removes exact duplicate rows, and drops rows missing required sensor or target fields. It retains sensor extremes rather than applying automatic IQR removal, because extremes may carry operational signal. EDA includes target prevalence, sensor distributions and failure-group comparisons, product type summaries, and correlation/visual summaries documented in the notebook.

The project uses a stratified 75/25 train/holdout split (`random_state=42`) before resampling. This yields 7,500 development rows and a 2,500-row final holdout with 85 failures. Scaling is fit on development data. Where SMOTE is used in cross-validation, it is inside an imbalanced-learn pipeline so each fold fits preprocessing/resampling only from that fold's training partition. The holdout is not resampled.

## 4. Models and experiments

### 4.1 Random Forest baseline and PCA comparison

The Phase 1 original-feature Random Forest uses a training-fitted `StandardScaler`, SMOTE (`random_state=42`), and a 300-tree Random Forest (`random_state=42`). Scaling is useful for SMOTE's distance calculation even though tree splits themselves do not require standardized inputs.

The retained PCA experiment standardizes the sensor pairs, reduces air/process temperature and speed/torque pairs to one component each, retains tool wear, and trains a second Random Forest. Scalers and PCA transformations are fitted on development data only. PCA preserves variance, not necessarily failure-discriminative information; it is a comparison, not the selected deployment model.

### 4.2 Cross-validation and additional models

Phase 2 compares Random Forest with SMOTE, `class_weight="balanced"`, and no resampling using five-fold stratified cross-validation and fold-local pipelines. Logistic Regression and HistGradientBoosting are additional justified comparisons. RandomizedSearchCV explores Random Forest estimator count, depth, split and leaf sizes, feature selection, and class weighting using average precision and stratified CV on development data. The final holdout is reserved for final evaluation.

### 4.3 Selected supervised model

The selected primary candidate is a pipeline of `StandardScaler`, SMOTE (`random_state=42`), and `HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=42)`. It uses the estimator's standard binary `predict()` decision. The returned `predict_proba` score is displayed as a model score; the project has not separately calibrated it, so it should not be interpreted as a well-calibrated real-world probability.

## 5. Evaluation methods

Evaluation reports accuracy, failure-class precision, recall, F1, ROC-AUC, average precision (called PR-AUC in project outputs), confusion matrices, false-positive rate (FP/(FP+TN)), and false-negative rate (FN/(FN+TP)). Failure-class metrics matter because normal rows greatly outnumber failure rows. Precision describes the fraction of alerts that match recorded failures; recall describes the fraction of recorded failures detected. PR-AUC summarizes precision-recall ranking over thresholds and is particularly informative under class imbalance. Accuracy alone is not an adequate selection criterion.

The Phase 2 calibration rule selected a threshold by maximizing F1 under a training-only calibration FPR cap; later Phase 3's deployed HGB rule is the documented default `predict()` threshold of 0.5. The final holdout is not used for fitting, preprocessing, feature selection, hyperparameter tuning, or threshold selection. HGB permutation importance is computed post hoc on holdout labels solely as descriptive reporting; it must not be fed back into model/feature selection.

## 6. Holdout results and interpretation

Metrics below are the actual saved results for the fixed 2,500-row holdout (85 failures); they are not newly estimated here.

| Model / policy | Precision | Recall | F1 | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting (primary) | 0.5726 | 0.8353 | 0.6794 | 0.8039 | 0.0219 | 0.1647 |
| Random Forest (Phase 1 baseline) | 0.5231 | 0.8000 | 0.6326 | 0.7103 | 0.0257 | 0.2000 |
| Isolation Forest (calibrated) | 0.1288 | 0.2000 | 0.1567 | 0.1385* | 0.0476 | 0.8000 |
| HistGradientBoosting OR Isolation Forest | 0.3133 | 0.8588 | 0.4591 | — | 0.0663 | 0.1412 |

*For Isolation Forest, failures are only a proxy label for evaluating anomaly alerts; this is not ground-truth anomaly detection performance. PR-AUC uses the continuous anomaly score. Hard OR policies have no continuous ranking score, so ROC-AUC/PR-AUC are not defined for them.*

HGB is retained as the primary model because it had the strongest failure recall and PR-AUC among the evaluated standalone supervised candidates at the documented operating point, with lower holdout FPR than the calibrated anomaly detector or combined OR policy. The RF baseline remains valuable and its holdout ROC-AUC was 0.9795 (HGB 0.9763), illustrating that model choice depends on the chosen operating metrics. The combined policy gains 2.35 percentage points of recall over HGB but reduces precision and raises FPR; it is not the default recommendation unless an operational process can tolerate more alerts.

## 7. Anomaly detection and condition assessment

The Isolation Forest (`n_estimators=300`, contamination 0.05, seed 42) is fitted on normal rows within a development-only sub-split of the 7,500-row training partition. Its scaler is fitted on those normal rows. Contamination and anomaly score threshold 0.571183 were selected using training-only calibration data under an FPR-constrained failure-proxy criterion. The anomaly score is `-score_samples`, where higher values indicate more unusual observations.

The three states are deterministic and hierarchical:

1. **FAILURE RISK:** the HGB classifier predicts failure.
2. **ANOMALOUS CONDITION:** HGB does not predict failure, but the IF score crosses its saved cutoff.
3. **NORMAL CONDITION:** neither signal raises an alert.

An anomalous condition means unusual relative to the detector's normal training observations; it is not a diagnosis. A false positive relative to `Machine failure` may be an unusual but non-failure sample, label mismatch, or proxy limitation. Sensor context and operational evidence are needed for interpretation.

## 8. Dashboard architecture

The local Streamlit application loads trusted joblib artifacts and metadata, validates feature schema/order, accepts one sensor snapshot or a CSV, and uses the same model pipelines and thresholds as the notebook. It includes overview, single prediction, batch prediction, and saved model analytics pages. Batch mode preserves input columns and marks invalid rows without scoring or silently imputing them. Analytics display saved holdout metrics rather than recalculating metrics on interactive inputs. See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md).

## 9. Limitations and future work

- AI4I is synthetic; performance does not establish transfer to a real plant, machine fleet, or changing operating regime.
- A random row split may not reflect time-forward or machine-group generalization. The dataset does not provide deployment-grade independent machine histories for this project.
- Positive examples are few and failures are heterogeneous; estimates have sampling uncertainty. Repeated external validation is needed.
- SMOTE interpolation may create physically implausible sensor combinations.
- HGB scores have not been separately calibrated; the fixed 0.5 classifier decision may not match maintenance costs.
- IF is evaluated against failure labels as a proxy, which conflates unusual operation with recorded failures.
- No cost model, operator feedback, drift monitoring, authenticated service, deployment infrastructure, or production safety validation is implemented.
- Feature importance and sensor percentile context are descriptive, not causal explanations. SHAP was not added.
- The system does not estimate RUL, confirm physical faults, or ingest live sensor streams.

Future work should use representative time/group-based validation, calibrated scores on an appropriate development calibration set, explicit maintenance costs and alert capacity, engineering review of synthetic resamples, and a real externally validated dataset before any operational use.

## 10. Conclusion

The project is a reproducible academic prototype that compares supervised failure classification, PCA-based reduction, and unsupervised anomaly scoring, then exposes the saved models through a local dashboard. On the fixed AI4I holdout, HGB offers the best balance for the selected recall/PR-AUC-focused objective among standalone models. The evidence supports an academic demonstration, not claims of production readiness or real-world predictive performance.
