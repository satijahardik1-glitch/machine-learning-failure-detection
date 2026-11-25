# Machine Fault Intelligence — Project Audit

> **Historical Phase 0 snapshot:** This report records the project before Phases 1–4 and is retained for audit history. Its statements that the notebook cannot execute, results are unavailable, and no application/dependency/documentation files exist were true at that snapshot and are superseded by the current README, Phase 5 documents, model artifacts, tests, and Streamlit application.

**Audit date:** 25 November 2026  
**Scope:** Read-only inspection of all files present in the project workspace and all 44 cells in `Machine_Fault_Intelligence_Improved.ipynb`. No training or notebook execution was performed; no model results are reported.

## Executive assessment

The workspace currently contains one notebook and one CSV nested in a dataset directory. The notebook is organized as a documented end-to-end experiment and describes several sound practices, including a stratified split before resampling, training-only scaler fitting for SMOTE, test-set preservation, and separation of supervised failure prediction from anomaly warnings.

The notebook is **not currently runnable from the project root as provided**. Its CSV path does not match the actual layout, and it references two Random Forest objects that are never initialized. The notebook has no saved cell outputs, and the workspace has no dependency manifest, run instructions, or application code. Therefore, there are no verified performance results to present. The project is a promising notebook prototype, but it is not yet ready for a reliable live demonstration or a results-based academic submission.

## 1. Existing project architecture

| Item | Observed state | Audit note |
|---|---|---|
| `Machine_Fault_Intelligence_Improved.ipynb` | 44 cells: markdown and Python code | Contains the complete described workflow, but execution is blocked as noted below. |
| `ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv` | 10,000 rows, 14 columns | The only data file present. The notebook expects `ai4i2020.csv` at the workspace root. |
| Python scripts/application | None found | No importable pipeline, command-line entry point, or user interface. |
| Dependency/environment files | None found | No `requirements.txt`, `pyproject.toml`, Conda file, or lockfile. |
| Documentation | No separate README/report found | Notebook markdown is the only project documentation. |
| Notebook execution outputs | None; all code cells show `execution_count=None` | No stored evidence of a successful end-to-end run or reported model results. |

Dataset inspection confirmed the expected sensor, target, identifier, type, and failure-mode columns. There are no missing values or exact duplicate rows in this CSV. The target counts are 9,661 normal and 339 failure rows (3.39% failure). These are dataset facts, not model performance results.

## 2. Current implementation status

| Objective | Status in notebook |
|---|---|
| AI4I data load and basic inspection | Implemented, but path is incorrect for current layout. |
| Basic cleaning and EDA | Implemented; missingness and duplicates are checked, then exact duplicates and required-field nulls are removed. |
| Leakage-aware feature exclusion | Failure-mode fields and identifiers are explicitly excluded; target leakage is avoided in the declared feature matrix. |
| Stratified train/test split | Implemented before SMOTE. |
| StandardScaler and SMOTE | Implemented on training data only; test data is transformed but not resampled. |
| Random Forest, raw sensor features | Evaluation code exists, but estimator is never constructed. |
| Pairwise PCA and Random Forest | Transformations are fitted on training rows only; estimator is never constructed. |
| Isolation Forest | Implemented and fitted on normal training rows; evaluated against failure labels as a proxy. |
| Combined assessment | Three-state labels and a binary OR alert are implemented. |
| Required metrics | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC and confusion matrices are implemented for Random Forest; combined metrics omit ROC-AUC/PR-AUC. FPR/FNR are not calculated. |
| Reproducibility and application | Not yet provided. |

## 3. Critical bugs and confirmed issues

### Confirmed runtime blockers

1. **Dataset path mismatch (cell 4).** The code calls `pd.read_csv("ai4i2020.csv")`, but the file is at `ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv`. Running from the workspace root raises a file-not-found error unless the CSV is manually moved or the working directory is changed.
2. **Undefined model variables (cells 25 and 30).** `rf_original.fit(...)` and `rf_pca.fit(...)` are called, but neither `rf_original` nor `rf_pca` is defined or initialized anywhere in the notebook. This raises `NameError` even after the CSV path is corrected.
3. **No executed outputs.** All notebook code cells have empty outputs and no execution counts. There is no recorded, inspectable successful run. This is a reproducibility/evidence gap rather than proof that every cell fails.

### Confirmed evaluation gaps

- **FPR and FNR are not reported.** The notebook prints FP and FN counts, but not false-positive rate `FP/(FP+TN)` or false-negative rate `FN/(FN+TP)`. Counts are useful but do not provide normalized rates.
- **Isolation Forest and combined alert lack ROC-AUC/PR-AUC.** The notebook reports a classification report and confusion matrix for these binary outputs. Anomaly scores from `decision_function` (with clear direction handling) could support ranking metrics for Isolation Forest. The combined OR rule is binary; AUC should only be supplied if a meaningful score/probability is defined, rather than pretending the hard alert is a calibrated score.
- **“Training accuracy” is printed for both Random Forests.** It is computed on original training rows that the models have already seen, after fitting on SMOTE-resampled data. This is a resubstitution diagnostic, not an independent generalization estimate, and can look optimistic. It should be labeled clearly or omitted from the final comparison.

## 4. Machine-learning methodology audit

### Data preparation and features

- The selected five sensor variables are a defensible baseline for the stated sensor-focused experiment. `Type` is analyzed descriptively but excluded from predictors. Consider a separately reported experiment including properly encoded Type if operating context is available at prediction time; preserve the five-feature baseline for comparability.
- Excluding `UDI`, `Product ID`, and the failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) is sensible for avoiding identifier memorization and target-derived information. This must remain explicit in any application input contract.
- The notebook only checks missingness for required features/target before dropping rows; the observed CSV has no missing values. Production code should validate numeric types, finite values, expected target labels, and inference column names rather than silently trusting data shape.
- Failure-mode labels and overall `Machine failure` are not perfectly interchangeable: this inspected CSV has 27 rows where the overall target differs from the row-wise maximum of the five mode flags. Do not replace the target with the mode flags or assume they exactly reconstruct it without investigating the dataset definition.

### Split, scaling, and SMOTE

- A stratified random split before SMOTE is a good baseline and the test set is left at its natural class balance. The random seed is explicitly set.
- Scaling is fit only on training rows before SMOTE, avoiding direct test-data leakage. Since the dataset has no missing values, fitting outside a pipeline does not currently introduce leakage. For cross-validation or a deployable workflow, put imputation/scaling/SMOTE/model steps in an `imblearn` pipeline so each fold fits transformations only on its own training fold.
- SMOTE creates interpolated minority observations and can make synthetic points in a small tabular dataset that do not correspond to physically feasible operating states. Compare against `class_weight="balanced"` and a no-resampling baseline; treat choice as a validation question, not an assumption.
- A single 75/25 split has sampling variability, particularly with only 339 positive rows. For a stronger academic estimate, use repeated stratified cross-validation for model/threshold selection and preserve a final untouched test set for one final report. If UDI encodes acquisition order or nearby rows are dependent, also discuss a group/time-aware split as a robustness check rather than assuming random rows represent future machines.

### Random Forest

- The forest hyperparameters are absent because the model objects are missing. Explicitly initialize both models, including `random_state`, and keep the experiment settings documented. Consider `n_jobs=-1` for execution convenience, and tune only within training folds.
- Accuracy is included but should not drive selection with a 3.39% positive class. Failure-class recall, precision, F1, PR-AUC, confusion-matrix counts, and FPR/FNR at an explicitly chosen operating threshold are more useful. The maintenance cost trade-off should determine the threshold.
- Default `predict()` threshold of 0.5 is not necessarily suitable. Select a threshold on validation predictions according to the stated false-alarm/missed-failure costs, then freeze it before test evaluation.
- A classification report and confusion matrix are appropriate; add uncertainty or repeated-fold variation where feasible. Avoid choosing the model based on the final test set repeatedly.

### PCA

- The PCA branch fits each scaler and PCA only on training data, then applies transformations to test rows; this avoids test leakage.
- This is a **custom pairwise compression from five features to three**, not conventional global PCA. The explanation is mostly clear, but using a single component in each pair discards half of each pair's directions. PCA maximizes input variance, not class separation. Compare scores before treating dimensionality reduction as beneficial.
- Tool wear is unscaled in the PCA feature frame, but the later `pca_representation_scaler` scales all three representation columns before SMOTE and prediction. This is internally consistent. The Random Forest itself does not need scaling, while SMOTE does.
- PCA comparison is fair at the same row split, but different feature representations and independently generated SMOTE points mean it compares two full training procedures, as intended. State that clearly.

### Isolation Forest and combined condition logic

- Fitting Isolation Forest on normal training observations only is a reasonable novelty-detection setup. Test labels are not used for fitting. StandardScaler is not generally required by Isolation Forest's axis-aligned tree splits; it is not a leakage problem here, but adds an unnecessary transformation and should be justified or removed in a future implementation.
- `contamination="auto"` fixes the algorithm's score threshold using its default rule, not the business cost or a target false-alarm rate. The number of anomalies and resulting false alarms should be reported, and threshold selection should be performed on training/validation normal data if an operational alert rate is needed.
- Failure labels are only a proxy for anomalies. Failures that do not look statistically unusual may be missed; valid unusual conditions may be flagged. Keep the notebook's caveat and describe Isolation Forest as a complementary warning signal.
- The three-state output is interpretable. The binary OR rule is easy to explain, but it may substantially increase false alarms. Report the combined alert's confusion matrix, precision, recall, F1, FPR, and FNR beside the primary model and compare against both baselines. It is an alert policy, not an independently trained model.

## 5. Evaluation coverage

| Measure | Current coverage | Recommendation |
|---|---|---|
| Accuracy | Yes, Random Forest and printed training diagnostic | Report test accuracy as secondary context; avoid train accuracy as model quality. |
| Precision | Yes, failure-class metric and classification report | Keep failure class explicit. |
| Recall | Yes, failure-class metric and classification report | Emphasize missed-failure implications. |
| F1-score | Yes, failure-class metric; used to select representation | Choose threshold/model using validation criteria and explain the cost trade-off. |
| ROC-AUC | Yes, Random Forest probability scores | Calculate on untouched test data only for final report. |
| PR-AUC / average precision | Yes, Random Forest probability scores | Appropriate for rare failure class; label as average precision if using sklearn's metric. |
| Confusion matrix | Yes, raw RF, PCA RF, Isolation Forest, combined | Include numeric counts in final submission. |
| False-positive rate | No, only FP count is printed | Add FP/(FP+TN), define positive class. |
| False-negative rate | No, only FN count is printed | Add FN/(FN+TP), define positive class. |

No metric values are claimed here because notebook execution was not performed and the saved notebook has no results.

## 6. Reproducibility and clean-environment readiness

**Current status: not reproducible from the supplied workspace.** The wrong data path and missing model initialization prevent a clean run. Beyond those blockers:

- Pin or bound Python and package versions. The notebook imports `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `imbalanced-learn`; exact versions are not declared. Compatibility of sklearn/imblearn APIs should be verified in the chosen environment.
- Add setup/run instructions, including the expected working directory and notebook kernel. A relative path should be resolved relative to the notebook/project root or configured explicitly.
- Keep seeds for split, SMOTE, PCA (where applicable), and both forests/Isolation Forest explicit. They are present for the split/SMOTE/PCA/Isolation Forest, but would need to be specified when initializing the forests.
- Decide whether to commit a cleaned notebook with outputs for presentation, or keep outputs cleared and provide a captured run artifact separately. The current saved file has no outputs.
- The notebook suppresses all warnings globally. This may hide useful compatibility or convergence warnings. Prefer targeted warning handling in a maintained version.

## 7. Application readiness

There is no application implementation. A small, dependable next step is to extract the notebook's chosen workflow into ordinary Python modules without introducing a framework until a user-facing need is settled:

1. A data-loading/validation function with a robust path and schema checks.
2. A training function that returns a fitted preprocessing/model pipeline and documented threshold.
3. A prediction function that accepts one or more sensor rows and returns RF failure score/prediction plus the distinct anomaly flag and condition status.
4. A saved model artifact and metadata containing feature order, package versions, threshold, training data reference, and model settings. Pickle/joblib artifacts should only be loaded from trusted sources.
5. A small CLI or notebook demonstration interface first; add a web UI only if the presentation requires live manual input.
6. Clear message that the output is a dataset-based decision aid, not a confirmed diagnosis or remaining-useful-life estimate.

The model needs an actual feature acquisition story: how the five sensor values are collected, units/ranges, missing/late readings, and how the application distinguishes a new observation from an out-of-distribution one.

## 8. Readiness assessment

| Use | Assessment | Reason |
|---|---|---|
| Academic submission | **Not ready yet** | Core method is documented, but the notebook cannot run as supplied and there are no verified results or declared environment. After fixes, include real outputs and methodological limitations. |
| Live demonstration | **Not ready yet** | No working clean-start notebook execution, fitted model initialization, or interactive prediction flow. |
| Technical viva | **Partially prepared** | Notebook markdown provides useful methodology and caveats. Be ready to explain SMOTE, rare-class metrics, PCA's variance objective, the proxy label for anomaly detection, and synthetic dataset limitations. Verify all claims with an actual run before presenting results. |
| Portfolio | **Not ready yet** | No README, dependency manifest, reproducible results, application entry point, or model artifact. |

## 9. Prioritized implementation roadmap

Complexity estimates are relative and assume this is a solo project. Estimates exclude grading/report formatting requirements that were not supplied.

| Phase | Priority | Work | Complexity |
|---|---:|---|---|
| 1. Make the notebook runnable | P0 | Correct CSV path; initialize both Random Forests; run cells top-to-bottom in a declared environment; resolve any observed runtime/API issues. | Low (1–2 hours) |
| 2. Establish credible evaluation | P0 | Add FPR/FNR, report all confusion-matrix counts and key test metrics, use validation for threshold/model decisions, compare SMOTE against class weighting/no resampling. | Medium (3–6 hours) |
| 3. Improve estimate reliability | P1 | Add repeated stratified cross-validation within training data; retain one final test set; assess threshold stability and false-alarm trade-offs. | Medium (4–8 hours) |
| 4. Reproducibility and documentation | P1 | Add dependency manifest, setup/run README, data path/schema notes, seeds/settings, limitations, and a clean verified run record. | Low–Medium (2–4 hours) |
| 5. Refactor for inference | P2 | Extract validation, fit, and predict code into modules/pipeline; store trusted artifact plus metadata; define input/output schema. | Medium (4–8 hours) |
| 6. Presentation/application layer | P2 | Build a simple CLI or presentation-ready input form, show probabilities/threshold/three-state condition, and explain model limitations. | Medium (4–10 hours), depending on UI scope |
| 7. Submission/portfolio polish | P3 | Final narrative, diagrams, screenshots, model comparison table based only on verified runs, and viva notes. | Low–Medium (2–5 hours) |

## 10. Files to modify after approval

No existing files were modified during this audit. The likely implementation targets are:

- `Machine_Fault_Intelligence_Improved.ipynb` — fix path, define models, improve metrics and evaluation narrative, and record verified outputs if desired.
- A new `requirements.txt` or `pyproject.toml` — declare the tested environment.
- A new `README.md` — clean setup/run steps, project explanation, and limitations.
- Optional new Python modules such as `src/data.py`, `src/train.py`, and `src/predict.py` — only if proceeding to an application-ready structure.
- Optional new application entry point — choose after deciding whether the demonstration needs CLI or UI.
- Dataset file — no change is indicated by this audit; its current schema matches the notebook's expected columns.

The report itself is a new file and does not alter project code, data, or notebook content.

## Audit boundary

This report is a static audit of the present files and notebook source. The notebook was not executed, no packages were installed, and no model was fit. Runtime issues reported above are confirmed from source/path inspection; unobserved environment-specific errors remain possible. Dataset shape/schema/count observations were read directly from the CSV. No model performance has been invented.
