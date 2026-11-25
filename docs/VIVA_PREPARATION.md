# Viva Preparation — Machine Fault Intelligence

Answers describe the implementation currently in this repository. They distinguish measured dataset results from claims that would require real equipment validation.

## 1. Project fundamentals

1. **What problem does the project address?** It evaluates five sensor readings for the AI4I `Machine failure` classification target and separately flags unusual operating patterns.
2. **What is the system output?** A failure-risk classifier result, a separate Isolation Forest anomaly signal, and one of three condition states.
3. **Does this system monitor a live machine?** No. It is a local Streamlit demonstration using user-entered snapshots or uploaded CSV rows.
4. **Does it predict remaining useful life?** No. It has no time-to-failure target, degradation trajectory, or RUL model.
5. **What is the main result?** On the fixed holdout, HGB had 0.8353 failure recall and 0.8039 PR-AUC at the saved decision rule.

## 2. Dataset and feature engineering

6. **What data is used?** The AI4I 2020 Predictive Maintenance Dataset from UCI, with 10,000 rows and a synthetic data process.
7. **What does the target mean?** `Machine failure` is the dataset's binary overall recorded-failure label.
8. **Which features are used?** Air temperature [K], process temperature [K], rotational speed [rpm], torque [Nm], and tool wear [min].
9. **Why exclude UDI and Product ID?** They are identifiers, not operating measurements, and could encourage memorization.
10. **Why exclude TWF, HDF, PWF, OSF, RNF?** They are failure-mode flags related to target construction and are not available as legitimate predictors for the intended input task.
11. **Why exclude Type from the primary model?** The central deployed contract is the same five-sensor experiment used in the comparisons. Type is used descriptively, not as a predictor.

## 3. Data preprocessing

12. **How are missing values handled?** Required columns are checked and rows missing required sensor/target values are dropped. The inspected local AI4I CSV has no missing values.
13. **Why keep outliers?** Extreme sensor readings may be relevant anomalies or failure signals, so there is no automatic IQR deletion.
14. **How is the data split?** A stratified 75/25 split with seed 42, producing 7,500 development and 2,500 holdout rows.
15. **What does stratification accomplish?** It approximately preserves the low failure prevalence in both partitions.
16. **Why scale before SMOTE?** SMOTE uses nearest-neighbor distances; scaling prevents large-unit features from dominating those distances.
17. **Does a tree model need scaling?** Tree split rules generally do not, but scaling is retained in these pipelines to support SMOTE consistently.

## 4. Machine learning algorithms

18. **How does Random Forest work?** It aggregates predictions from multiple randomized decision trees to reduce variance relative to a single tree.
19. **Why add HistGradientBoosting?** It is a distinct boosting approach and was a strong additional candidate in the project's CV and holdout evaluation.
20. **What is Logistic Regression's role?** It provides a simpler linear baseline whose performance can be compared with tree ensembles.
21. **Why not just use accuracy?** With only 3.39% failures, predicting normal for every case would still yield high accuracy while detecting no failures.
22. **Why retain the PCA experiment?** It measures whether pairwise dimensionality reduction preserves predictive information and preserves the earlier project objective.

## 5. Class imbalance and SMOTE

23. **What is class imbalance here?** 339 of 10,000 observations are failures, about 3.39%.
24. **What does SMOTE do?** It creates synthetic minority examples by interpolating among minority-class neighbors.
25. **How is SMOTE leakage prevented?** It is included inside imbalanced-learn pipelines, so it is fit only on each fold's training portion and never on the holdout.
26. **What are SMOTE's limitations?** Interpolated sensor combinations may be physically implausible, and oversampling does not create new independent failures.
27. **What alternatives were compared?** Random Forest with class weighting and Random Forest without resampling, in addition to SMOTE.

## 6. PCA

28. **What does PCA optimize?** It finds orthogonal directions that explain variance in the input data, without using target labels.
29. **Why scale before PCA?** The sensor variables have different units and numerical scales; without scaling, high-range values can dominate components.
30. **How is PCA leakage avoided?** Scalers and PCA are fitted on training rows and then applied unchanged to holdout rows.
31. **Why can PCA hurt classification?** High-variance directions need not be the directions most useful for distinguishing the minority failure class.

## 7. Evaluation metrics

32. **Define precision.** TP/(TP+FP): among predicted failures, the fraction that are recorded failures.
33. **Define recall.** TP/(TP+FN): among recorded failures, the fraction detected.
34. **Define F1.** The harmonic mean of precision and recall; it is low if either is low.
35. **Define FPR and FNR.** FPR=FP/(FP+TN); FNR=FN/(FN+TP). They quantify false alarms among non-failures and misses among failures.
36. **What is PR-AUC here?** Average precision from the precision-recall curve over score thresholds. It summarizes positive-class ranking and is informative for rare failures.
37. **Why also report ROC-AUC?** It measures ranking discrimination across true-positive and false-positive rates; it complements PR-AUC but can look favorable under strong imbalance.
38. **What is the HGB holdout confusion matrix?** TN=2362, FP=53, FN=14, TP=71, for 2,500 observations.
39. **Why is RF selected as a baseline despite HGB's stronger recall?** RF is the retained Phase 1 original-feature comparison; HGB is the primary model under the project's recall/PR-AUC-focused objective. RF had a slightly higher ROC-AUC (0.9795 vs 0.9763), so selection depends on operating priorities.

## 8. Hyperparameter tuning and cross-validation

40. **Why use StratifiedKFold?** Each fold keeps an approximate failure proportion, making fold comparisons less sensitive to class composition.
41. **Why put preprocessing in the CV pipeline?** Every fold learns scaling and resampling only from its training fold, preventing validation-fold information from leaking into fit steps.
42. **What does RandomizedSearchCV optimize?** It samples combinations of forest hyperparameters and scores candidates by average precision using stratified CV on development data.
43. **Why must the holdout stay untouched?** Repeatedly selecting models or thresholds from holdout results makes reported generalization optimistic.
44. **How was threshold choice constrained?** A training-only calibration process selected operating points under a stated false-positive-rate cap; the deployment HGB classifier uses its documented standard prediction rule.

## 9. Isolation Forest and anomaly detection

45. **How does Isolation Forest detect anomalies?** It isolates observations through randomized partition trees; observations requiring shorter isolation paths receive more anomaly-like scores.
46. **Is it trained on failure labels?** No. It is fitted on normal sensor rows from a development-only training sub-split.
47. **Why call failure metrics a proxy?** The dataset does not provide independent anomaly ground truth, so recorded failure labels are only a comparison target, not a definition of anomaly.
48. **What do contamination and the score cutoff mean?** Contamination sets an internal expected outlier fraction; the project separately selects a score threshold using training-only calibration. Its score is `-score_samples`, with larger values more unusual.
49. **What is the saved IF holdout result?** Precision 0.1288, recall 0.2000, F1 0.1567, PR-AUC 0.1385, and FPR 0.0476 against failure labels as proxy.
50. **Why not combine every IF alert with HGB?** OR logic raises recall to 0.8588 but drops precision to 0.3133 and raises FPR to 0.0663. The added alerts may not be operationally acceptable.
51. **What can an IF false positive mean?** It is anomalous to the learned normal distribution but has no recorded failure; it may be a rare valid regime, a label limitation, or an unusual condition worth review.

## 10. Model deployment and Streamlit

52. **What artifacts are loaded?** Trusted local joblib pipelines for HGB, Random Forest, and Isolation Forest, plus JSON metadata and saved evaluation results.
53. **How is feature order enforced?** Metadata stores the five-feature order; artifact loading verifies it, and inference selects sensor columns in that order.
54. **How does batch validation behave?** It requires all sensor columns, rejects duplicate/reserved headers, validates numeric finite values and basic ranges, preserves invalid rows with messages, and does not score them.
55. **Does interactive input alter evaluation metrics?** No. The analytics page reads saved evaluation JSON; user predictions are separate from the final evaluation.
56. **Are displayed HGB scores calibrated probabilities?** No. They are `predict_proba` outputs without a separate calibration step and should be interpreted cautiously.

## 11. Limitations and future scope

57. **What is the most important limitation?** AI4I is synthetic and performance has not been validated on real plant data.
58. **What split limitation remains?** A random row split does not establish performance on future time periods or unseen machines.
59. **Do feature importance values prove causality?** No. They describe model behavior under a chosen importance method and dataset; they do not establish causal effects.
60. **What would be required before deployment?** Representative real data, time/group-based external validation, calibrated operating thresholds tied to maintenance costs, drift and safety processes, and engineering oversight.
61. **What is a sensible next academic improvement?** Evaluate generalization with an external or grouped/time-ordered dataset and quantify uncertainty without using the final holdout to make further choices.

## Difficult examiner prompts

### Could preprocessing be leaked even if the holdout is untouched?

Yes. Fitting a scaler, resampler, feature selector, or imputer before cross-validation would expose validation-fold distribution information to training. The Phase 2 imbalanced-learn pipeline keeps scaling and SMOTE inside each training fold. The final artifact fit uses only the fixed training partition.

### Why can high recall still be an undesirable operating point?

High recall reduces missed failures, but it can produce many false alarms. At low prevalence, even a modest FPR may create more false than true alerts. The intended maintenance workflow must set an acceptable alert burden and missed-failure cost before choosing a deployment threshold.

### Why is PR-AUC more informative than accuracy for this dataset?

The positive class is rare. Accuracy is dominated by normal observations, while the precision-recall curve focuses on failure retrieval and the reliability of positive alerts. PR-AUC's baseline depends on prevalence, so it should be interpreted with class distribution and holdout context.

### Does an Isolation Forest “false positive” prove a bad detector?

No. It is a disagreement with the recorded failure label. Since the detector learns unusualness rather than failure, an alert can indicate a rare but valid operating point. Conversely, the failure label is not ground truth for all anomalies.

### Why choose HGB if RF ROC-AUC is slightly higher?

The stated objective prioritizes failure recall and PR-AUC while controlling false alarms. HGB has higher holdout recall and PR-AUC, while its FPR is lower than RF's in the saved results. RF's slightly higher ROC-AUC is a meaningful trade-off and is documented rather than hidden.

### Why not choose the combined HGB OR IF model because it catches more failures?

It detects two additional holdout failures compared with HGB (73 vs 71) but creates 107 additional false positives (160 vs 53). The 0.0235 recall increase comes with substantially lower precision and higher FPR; the project therefore keeps HGB standalone as the default.
