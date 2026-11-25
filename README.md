# Machine Fault Intelligence

A reproducible notebook project for predictive maintenance using the AI4I 2020 Predictive Maintenance Dataset. The existing experiments compare a Random Forest on five sensor features with a Random Forest on pairwise PCA features, apply SMOTE to training data only, and evaluate Isolation Forest as a complementary anomaly warning.

## Project files

- `Machine_Fault_Intelligence_Improved.ipynb` — analysis, training, evaluation, and combined condition assessment.
- `ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv` — source dataset.
- `requirements.txt` — pinned Python dependencies.

## Requirements

- Python 3.11 (the dependency versions in `requirements.txt` were selected and verified with Python 3.11.9).
- Run commands from the project root, the directory containing this README and notebook.

## Installation

Create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv_clean
.venv_clean\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements install also installs the project package in editable mode, so Streamlit can resolve `app` even when it runs the entrypoint from the `app` directory. The existing `.venv` is left untouched. If you need a fresh environment, use the separate `.venv_clean` path above rather than reusing a broken environment.

If PowerShell blocks environment activation, use the interpreter directly instead:

```powershell
.venv_clean\Scripts\python.exe -m pip install -r requirements.txt
.venv_clean\Scripts\python.exe -m jupyter lab
```

## Run the notebook

From the project root:

```powershell
jupyter lab
```

Open `Machine_Fault_Intelligence_Improved.ipynb`, select the Python 3 kernel, and choose **Restart Kernel and Run All Cells**. The notebook reads the dataset from the checked-in relative path. Its outputs are saved in the notebook after a successful full run.

To execute headlessly from the project root with the installed Jupyter dependencies:

```powershell
jupyter nbconvert --to notebook --execute Machine_Fault_Intelligence_Improved.ipynb --inplace
```

The dataset is not resampled before splitting: stratified train/test split happens first, and SMOTE is applied only to the training representations. The holdout test rows retain their original class distribution. The notebook reports actual metrics from the run; do not compare or present numbers from a different data split as though they came from this saved run.

## Reproducibility notes

- The notebook sets `RANDOM_STATE = 42` and uses it in the split, SMOTE, both Random Forest models, PCA configuration, and Isolation Forest.
- The saved notebook contains outputs from its latest successful execution. To reproduce them, restart the kernel and run every cell in order.
- The dataset's `Machine failure` label is the supervised target. Failure-mode flags, identifiers, and product ID are not used as predictors.
- Isolation Forest anomaly flags are a complementary unusual-condition signal, not a diagnosis of a specific failure.
- AI4I 2020 is a synthetic dataset. Performance on this dataset does not establish performance on a live factory or support remaining-useful-life estimates.


## Phase 2 evaluation additions

The notebook preserves the Phase 1 Random Forest, PCA, Isolation Forest, and combined-alert experiments and adds:

- Five-fold `StratifiedKFold` comparisons of Random Forest with SMOTE, `class_weight='balanced'`, and no resampling. Scaling and SMOTE are inside an imbalanced-learn pipeline and are refit within each training fold.
- Logistic Regression and HistGradientBoosting comparisons with fold-local preprocessing.
- A 16-candidate `RandomizedSearchCV` over forest size, depth, split/leaf sizes, max features, and class weight, scored by average precision. Search uses a development slice of the training partition.
- A separate stratified threshold-calibration slice from the training partition. The selected threshold maximizes F1 subject to calibration FPR <= 5%; final holdout labels are not used for tuning or threshold choice.
- Holdout classification reports, confusion matrices, ROC and precision-recall curves, false-positive/false-negative rates, and standalone/combined anomaly-alert comparison.

On the saved run, HistGradientBoosting + SMOTE is the higher-recall candidate under the chosen mean-CV FPR cap: CV recall 0.783 (SD 0.080), FPR 0.0235 (SD 0.0046), F1 0.6388 (SD 0.0548), and PR-AUC 0.7713 (SD 0.0694). Holdout recall was 0.8353, precision 0.5726, F1 0.6794, PR-AUC 0.8039, and FPR 0.0219. The no-resampling Random Forest remains a lower-alarm alternative with holdout recall 0.6706, precision 0.8769, F1 0.7600, and FPR 0.0033. These are results from this dataset and split, not estimates of real-factory performance.


## Phase 3 anomaly and condition analysis

Phase 3 preserves the previous notebook sections and adds a training-only Isolation Forest development/calibration split. The detector is fitted only on normal development rows. Contamination candidates are evaluated on the separate calibration slice against `Machine failure` as a proxy label; the final holdout is not used to choose contamination or the anomaly score threshold. The chosen configuration maximizes calibration F1 subject to calibration FPR <= 5%.

In the saved run, the selected contamination was `0.05`, with score threshold `0.571183`. On the final holdout, Isolation Forest detected 20.0% of recorded failures at 12.88% precision and 4.76% FPR (failure-proxy PR-AUC 0.1385). Its score ranking was constant across contamination candidates because contamination changes the cutoff rather than the seeded forest's ranking.

The standalone HistGradientBoosting model achieved holdout precision 0.5726, recall 0.8353, F1 0.6794, PR-AUC 0.8039, and FPR 0.0219. HistGradientBoosting OR calibrated Isolation Forest increased recall to 0.8588 but reduced precision to 0.3133 and increased FPR to 0.0663. The standalone model is the recommended default; the combined alert is only useful if the maintenance process accepts its additional false alarms.

The notebook also investigates non-failure anomaly flags, plots score distributions and operating points, and constructs FAILURE RISK / ANOMALOUS CONDITION / NORMAL CONDITION states. SHAP was not installed in the verified environment; global impurity/permutation importance and sensor-context descriptions are used instead. Anomaly flags indicate unusual behavior, not confirmed failure, and this project does not estimate remaining useful life.

## Phase 4 interactive dashboard

The Streamlit application adds an interactive interface without moving or rewriting the notebook or source CSV:

```text
app/
  main.py                 Navigation, theme, shared model/data loading
  inference.py            Artifact loading, input validation, batch inference, condition rules
  pages/overview.py       Historical dataset and saved holdout overview
  pages/predict.py        Single sensor snapshot prediction
  pages/batch.py          CSV validation, batch predictions, CSV download
  pages/analytics.py     Saved evaluation metrics and interactive model charts
scripts/
  train_artifacts.py      Reproducibly train local artifacts from development rows only
models/
  hist_gradient_boosting_pipeline.joblib
  random_forest_pipeline.joblib
  isolation_forest_pipeline.joblib
  metadata.json
  evaluation_results.json
.streamlit/config.toml    Dark industrial theme and upload size limit
```

### Dashboard setup and launch

From the project root, install the pinned dependencies and create the local model artifacts:

```powershell
python -m venv .venv_clean
.venv_clean\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/train_artifacts.py
python -m streamlit run app/main.py
```

Open the local URL printed by Streamlit. The app does not download model or data files. If an artifact is missing, regenerate it with the training script. The local `.joblib` files are trusted project artifacts; do not load replacements from untrusted sources.

### Model integration

- **Primary:** HistGradientBoosting pipeline (`StandardScaler` → SMOTE → `HistGradientBoostingClassifier`, 150 iterations, learning rate 0.1, seed 42), trained on the notebook's fixed 7,500-row development partition. Failure decision uses the existing model prediction threshold.
- **Comparison:** Phase 1 Random Forest pipeline (`StandardScaler` → SMOTE → 300-tree Random Forest, seed 42), trained on the same development partition.
- **Anomaly:** Isolation Forest pipeline, trained on normal rows from the notebook's Phase 3 detector-development subset, with contamination 0.05 and the saved anomaly-score cutoff 0.571183. The score is `-score_samples`; larger means more unusual.
- The feature order is stored and checked at load time. The final 2,500-row evaluation holdout is excluded from artifact fitting.
- The metrics page displays saved notebook evaluation results; prediction inputs never recalculate or overwrite them. HGB model outputs are scores, not separately calibrated probabilities.

### Pages and behavior

- **Overview:** historical dataset counts, the saved holdout condition distribution, primary model metrics, and latest prediction in the current session. Labels distinguish these offline results from live monitoring.
- **Single prediction:** five unit-labelled sensor inputs with dataset-median defaults, HGB result, Random Forest comparison, anomaly score, and the three-state assessment.
- **Batch prediction:** upload a CSV with all required sensor columns, retain extra/input columns, mark invalid rows without silently imputing, summarize condition states, and download prediction results.
- **Model analytics:** saved accuracy, precision, recall, F1, ROC-AUC and PR-AUC where available; confusion matrices, sensor importance, false-alarm/missed-failure comparison, and the saved contamination calibration results.

The anomaly flag is not a machine-failure diagnosis. The three states are FAILURE RISK (supervised classifier predicts failure), ANOMALOUS CONDITION (no failure prediction but anomaly score crosses the saved cutoff), and NORMAL CONDITION. The app does not claim live industrial monitoring or remaining-useful-life prediction.

### Phase 4 validation

The saved artifacts were rebuilt from the fixed training split and reproduced the notebook holdout predictions: HGB precision 0.5726, recall 0.8353, F1 0.6794, PR-AUC 0.8039; Phase 1 Random Forest precision 0.5231, recall 0.8000, F1 0.6326, PR-AUC 0.7103; calibrated Isolation Forest precision 0.1288, recall 0.2000, F1 0.1567, PR-AUC 0.1385. The final holdout was evaluated only for verification and was not used to fit any artifact.

Inference, batch, and dashboard-startup tests are in the `tests/` directory and can be run with:

```powershell
python -m unittest discover -v
```
