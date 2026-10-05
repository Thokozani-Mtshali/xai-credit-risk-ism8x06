# Modelling Log
*(Covers Step 3 onward — baseline, RF/XGBoost/ensemble, SMOTE comparisons, SHAP. Kept separate from Preprocessing_Decision_Log.md, which covers Steps 1-2 only.)*

---

## Step 3 — Baseline Logistic Regression

Notebook: `04_baseline_logistic_regression.ipynb`. Reloaded scaled train/test splits from Step 2; `.squeeze()` used to restore y files to Series after CSV round-trip.

`LogisticRegression(max_iter=1000, random_state=42)` fit per dataset. Results:

| Dataset | AUC-ROC | F1 (minority) | G-mean | MCC |
|---|---|---|---|---|
| German Credit | 0.805 | 0.544 | 0.645 | 0.401 |
| Home Credit | 0.744 | 0.020 | 0.100 | 0.062 |
| LendingClub | 0.853 | 0.200 | 0.333 | 0.331 |

**Key finding:** AUC alone looks acceptable across all three, but F1/G-mean/MCC collapse on Home Credit and LendingClub — the two more severely imbalanced datasets. Verified this is a genuine imbalance effect, not a bug: on Home Credit's 61,502-row test set (true positive rate ~8%), the model predicted positive for only 101 rows (0.16%). This directly validates the proposal's decision (Section 5) to report F1/G-mean/MCC alongside AUC rather than AUC/accuracy alone, and provides baseline evidence for RQ2.

**Literature check:** a comparable 2025 study (same 5-model design: LR/RF/XGBoost/stacking/MLP) found the same qualitative pattern — LR shows high recall/low precision, RF achieves best F1 (0.81). A separate ensemble-comparison study reported baseline AUC ~0.66–0.69, F1 ~0.18–0.33 — German Credit and LendingClub AUC here exceed that range; Home Credit's F1 (0.02) is more severe than anything in that comparison set. Li & Chen (2020)'s exact figures weren't found in open search, but their abstract's conclusion (LR beats other baselines on most metrics, ensembles beat individual learners overall) is consistent with this result.

Logged as Early Finding #1 (candidate — pending CV confirmation and RF/XGBoost comparison). Full detail in `Early_Findings_Log.md`.

---

## Step 4 — RF, XGBoost, Stacked Ensemble, With/Without SMOTE

Notebook: `05_ensemble_models.ipynb`.

**Bug fixed:** XGBoost rejects feature names containing `[`, `]`, `<` (raised `ValueError: feature_names must be string, and may not contain [, ] or <`). Occurred because German Credit's one-hot columns include labels like `"< 0 DM"`. Fixed with a `clean_column_names()` helper (regex-replaces `[`, `]`, `<`, `>` with `_`) applied to all three datasets' train/test X before any modelling. RF was unaffected (only XGBoost enforces this), which is why the error appeared only when XGBoost was introduced.

All models used `random_state=42`. SMOTE applied via `imblearn.pipeline.Pipeline` (not sklearn's) to ensure it only resamples training folds/data, never test data. Stacking used `RF + XGBoost` as base learners, `LogisticRegression` as meta-learner, `StratifiedKFold(5)` internally.

**Results — German Credit:**
| Config | AUC-ROC | F1 | G-mean | MCC |
|---|---|---|---|---|
| RF (no SMOTE) | 0.796 | 0.525 | 0.627 | 0.394 |
| RF (SMOTE) | 0.784 | 0.561 | 0.663 | 0.409 |
| XGBoost (no SMOTE) | 0.778 | 0.463 | 0.590 | 0.271 |
| XGBoost (SMOTE) | 0.795 | 0.584 | 0.687 | 0.423 |
| Stacked (no SMOTE) | 0.786 | 0.500 | 0.610 | 0.355 |
| **Stacked (SMOTE)** | 0.790 | **0.636** | **0.722** | **0.504** |

**Results — Home Credit:**
| Config | AUC-ROC | F1 | G-mean | MCC |
|---|---|---|---|---|
| RF (no SMOTE) | 0.724 | 0.002 | 0.028 | 0.024 |
| RF (SMOTE) | 0.714 | 0.016 | 0.090 | 0.039 |
| XGBoost (no SMOTE) | 0.746 | 0.059 | 0.178 | 0.109 |
| XGBoost (SMOTE) | 0.746 | 0.062 | 0.182 | 0.116 |
| Stacked (no SMOTE) | 0.744 | 0.087 | 0.218 | **0.132** |
| Stacked (SMOTE) | 0.722 | **0.102** | **0.245** | 0.111 |

**Results — LendingClub:**
| Config | AUC-ROC | F1 | G-mean | MCC |
|---|---|---|---|---|
| RF (no SMOTE) | 0.851 | 0.286 | 0.408 | 0.405 |
| RF (SMOTE) | 0.816 | **0.000** | **0.000** | **0.000** |
| **XGBoost (no SMOTE)** | **0.907** | **0.468** | 0.553 | **0.549** |
| XGBoost (SMOTE) | 0.885 | 0.440 | 0.552 | 0.485 |
| Stacked (no SMOTE) | 0.868 | 0.468 | 0.553 | 0.549 |
| Stacked (SMOTE) | 0.827 | 0.318 | 0.441 | 0.408 |

**Headline finding (RQ3):** SMOTE's benefit is inversely related to imbalance severity/minority sample count — helps broadly on German Credit (mild imbalance, plenty of minority samples), mixed on Home Credit (severe imbalance, large dataset), actively harms every model on LendingClub (most severe imbalance, only ~142 minority training samples — RF collapses completely to F1/G-mean/MCC=0.000). Confirmed via diagnostic re-run: RF predicted 0/2000 positive cases after SMOTE on LendingClub; minority count was well above SMOTE's k_neighbors=5 requirement, ruling out a technical failure — points instead to RF being more sensitive than boosting models to SMOTE's synthetic samples on very sparse minority classes.

**Headline finding (RQ1):** stacked ensemble was the best configuration on German Credit and Home Credit, but plain XGBoost (no SMOTE) outperformed stacking on LendingClub — ensemble/stacking advantage is real but not universal across datasets.

Both logged as Early Finding #2 (status: confirmed, pending CV cross-check). Full detail and literature comparison in `Early_Findings_Log.md`.

---

## Step 6 — SHAP Analysis (RQ4)

Notebook: `06_shap_analysis.ipynb`. XGBoost + TreeExplainer used throughout (chosen over explaining the full stack, since TreeExplainer requires a tree-based model and XGBoost alone performed close to the stack).

**German Credit:** global summary + single-row waterfall plot generated. Top feature: `status_checking_account = "no checking account"` (counter-intuitive direction — associated with lower predicted risk). SHAP stability check (SMOTE vs no-SMOTE, Spearman on mean |SHAP|): **0.983** — highly stable.

**Home Credit:** attempted, hit `MemoryError` fitting/explaining two XGBoost models on 246,005×100 training data, even after reducing SHAP evaluation sample to 1,000 rows. Consistent with the approved plan's Risk 4 (anticipated SHAP compute risk on this dataset). Not completed — flagged as a limitation in `Findings_Section.md`, with Kaggle Notebooks suggested as the fix (more RAM, hosts this dataset natively).

**LendingClub:** completed successfully (small dataset, no memory issue). SHAP stability check: **Spearman = 0.797** — clearly positive but noticeably lower than German Credit's 0.983. Top features overlap substantially (`paid_total`, `installment`, `paid_principal`, two `issue_month` categories in both top-10s) but exact ranking shifts more than German Credit did.

**Cross-dataset synthesis (2 of 3 datasets):** SHAP stability under SMOTE appears to track with SMOTE's effect on predictive performance — high stability where SMOTE helped (German Credit), lower stability where SMOTE hurt (LendingClub, per Step 4 findings). Home Credit, where SMOTE's effect was "mixed," is predicted to fall between 0.797–0.983 if this pattern holds — untested due to memory constraint.

**Home Credit resolved on second attempt** (locally, no cloud environment needed): the earlier `MemoryError` was fixed by fitting/explaining one model at a time — fit no-SMOTE model → SHAP → extract ranking → `del` + `gc.collect()` → then fit SMOTE model → repeat — rather than holding both fitted models and both SHAP arrays in memory simultaneously. Also reduced SHAP sample to 300 rows (from the earlier attempted 1,000). Result: **Spearman = 0.906.**

**Final three-dataset RQ4 result:**
| Dataset | Spearman (SMOTE vs no-SMOTE SHAP ranking) | SMOTE's performance effect (Step 4) |
|---|---|---|
| German Credit | 0.983 | Clearly positive |
| Home Credit | 0.906 | Mixed |
| LendingClub | 0.797 | Clearly negative |

**Core finding:** SHAP stability under SMOTE tracks monotonically with SMOTE's effect on predictive performance across all three datasets — not a fixed property of the technique, but conditional on how well SMOTE works for that specific dataset. This directly links RQ3 and RQ4 into a single coherent result rather than two separate answers.

Logged as Early Finding #6 (status: confirmed, 3/3 datasets, clean monotonic pattern). Full detail in `Early_Findings_Log.md`. Findings Section and this log both updated to reflect the complete three-dataset picture.

---

## Step 7 — Cross-Validation Confirmation

Notebook: `07_cv_confirmation.ipynb`. Reused the 5 fold assignments already saved in Step 2 (`logs/fold_assignments/lending_folds.json`) rather than regenerating new folds, keeping this traceable back to the original audit trail.

**LendingClub, RF with/without SMOTE, all 5 folds:**
| Fold | Config | AUC-ROC | F1 | G-mean | MCC | Positive predictions |
|---|---|---|---|---|---|---|
| fold_0 | No SMOTE | 0.750 | 0.194 | 0.327 | 0.325 | 3 |
| fold_0 | SMOTE | 0.787 | 0.188 | 0.327 | 0.280 | 4 |
| fold_1 | No SMOTE | 0.760 | 0.133 | 0.267 | 0.265 | 2 |
| fold_1 | SMOTE | 0.805 | **0.000** | **0.000** | -0.003 | **0** |
| fold_2 | No SMOTE | 0.784 | 0.069 | 0.189 | 0.187 | 1 |
| fold_2 | SMOTE | 0.841 | **0.000** | **0.000** | **0.000** | **0** |
| fold_3 | No SMOTE | 0.727 | 0.067 | 0.186 | 0.184 | 1 |
| fold_3 | SMOTE | 0.791 | **0.000** | **0.000** | **0.000** | **0** |
| fold_4 | No SMOTE | 0.842 | 0.129 | 0.263 | 0.260 | 2 |
| fold_4 | SMOTE | 0.828 | **0.000** | **0.000** | -0.003 | 1 |

**Result: the RF+SMOTE collapse is confirmed as a genuine, consistent pattern — not a single-split fluke.** Exactly 0 positive predictions in 3/5 folds, only 1-4 in the remaining 2 (vs. hundreds of true positives per fold). RF without SMOTE was already weak (F1 0.067-0.194) but SMOTE made it worse in every single fold, no exceptions.

This upgrades the LendingClub component of Early Finding #2 from "single-split" to "5-fold CV confirmed" — now the most rigorously validated result in the study. `Findings_Section.md` and `Early_Findings_Log.md` (entry #7) both updated accordingly.

**Superseded by Step 10 below** — all single-split SMOTE comparisons across all three datasets have now been replaced with tuned, cross-validated results.

---

## Step 10 — Final Hyperparameter Tuning and Full Cross-Validation (closes methodology gaps against approved proposal Sections 7.3, 7.5)

Notebook: `10_final_tuning_cv.ipynb`. Grid search (`GridSearchCV`, German Credit and LendingClub) / randomized search (`RandomizedSearchCV`, Home Credit — see scoping note below) over the exact parameter grids specified in the approved proposal: RF (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`), XGBoost (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`). Scoring metric: F1 (minority class), not AUC, given the whole study's focus on minority detection. All final results evaluated via full cross-validation (`cross_validate`), reporting mean ± standard deviation across folds, rather than a single train/test split.

**German Credit (5-fold CV throughout):**
| Config | AUC-ROC | F1 | G-mean | MCC |
|---|---|---|---|---|
| RF (no resample) | 0.790±0.039 | 0.481±0.103 | 0.589±0.083 | 0.357±0.114 |
| RF (SMOTE) | 0.791±0.042 | 0.591±0.091 | 0.694±0.073 | 0.430±0.116 |
| RF (SMOTE-ENN) | 0.778±0.044 | 0.602±0.029 | 0.713±0.026 | 0.397±0.050 |
| XGBoost (no resample) | 0.785±0.041 | 0.579±0.066 | 0.679±0.050 | 0.424±0.090 |
| XGBoost (SMOTE) | 0.772±0.037 | 0.607±0.050 | 0.720±0.043 | 0.416±0.073 |
| XGBoost (SMOTE-ENN) | 0.778±0.046 | 0.610±0.036 | 0.722±0.032 | 0.412±0.060 |
| Stacked (no resample) | 0.796±0.039 | 0.500±0.082 | 0.605±0.067 | 0.372±0.091 |
| Stacked (SMOTE) | 0.794±0.044 | 0.582±0.083 | 0.689±0.068 | 0.410±0.106 |
| Stacked (SMOTE-ENN) | 0.786±0.044 | 0.600±0.037 | 0.714±0.032 | 0.397±0.060 |

**LendingClub (5-fold CV throughout):**
| Config | AUC-ROC | F1 | G-mean | MCC |
|---|---|---|---|---|
| RF (no resample) | 0.772±0.047 | 0.118±0.047 | 0.246±0.053 | 0.244±0.053 |
| RF (SMOTE) | 0.769±0.010 | 0.135±0.096 | 0.352±0.148 | 0.121±0.099 |
| RF (SMOTE-ENN) | 0.785±0.014 | 0.147±0.104 | 0.302±0.120 | 0.152±0.108 |
| XGBoost (no resample) | 0.873±0.040 | 0.403±0.116 | 0.504±0.092 | 0.486±0.100 |
| XGBoost (SMOTE) | 0.858±0.036 | 0.389±0.148 | 0.497±0.124 | 0.455±0.127 |
| XGBoost (SMOTE-ENN) | 0.850±0.028 | 0.378±0.140 | 0.491±0.119 | 0.447±0.114 |
| Stacked (no resample) | 0.835±0.050 | 0.365±0.132 | 0.471±0.108 | 0.454±0.116 |
| Stacked (SMOTE) | 0.798±0.023 | 0.361±0.179 | 0.481±0.156 | 0.407±0.167 |
| Stacked (SMOTE-ENN) | 0.808±0.021 | 0.329±0.134 | 0.460±0.122 | 0.376±0.122 |

**Home Credit (3-fold CV throughout — scoped down; see note below):**
| Config | AUC-ROC | F1 | G-mean | MCC |
|---|---|---|---|---|
| RF (no resample) | 0.726±0.002 | 0.001±0.000 | 0.019±0.002 | 0.015±0.001 |
| XGBoost (no resample) | 0.746±0.001 | 0.041±0.002 | 0.146±0.003 | 0.087±0.001 |
| Stacked (no resample) | 0.744±0.002 | 0.076±0.005 | 0.203±0.007 | 0.118±0.009 |
| RF (SMOTE) | 0.675±0.002 | 0.205±0.004 | 0.468±0.007 | 0.125±0.004 |
| XGBoost (SMOTE) | 0.674±0.003 | 0.190±0.005 | 0.434±0.009 | 0.113±0.005 |
| Stacked (SMOTE) | 0.680±0.003 | 0.209±0.004 | 0.479±0.004 | 0.128±0.005 |
| RF (SMOTE-ENN) | 0.706±0.003 | 0.245±0.003 | 0.546±0.001 | 0.166±0.003 |
| XGBoost (SMOTE-ENN) | 0.705±0.003 | 0.242±0.003 | 0.536±0.009 | 0.163±0.004 |
| Stacked (SMOTE-ENN) | 0.711±0.003 | 0.245±0.002 | 0.533±0.001 | 0.167±0.002 |

**Scoping note (Home Credit):** the original plan (20 search iterations, 5-fold CV, matching German Credit/LendingClub) was attempted but one search (RF+SMOTE-ENN) exceeded 14 hours without completing, at which point it was interrupted. All remaining Home Credit searches and final CV evaluations were rescoped to `RandomizedSearchCV` with 5 iterations and 3-fold CV. This is a real, documented asymmetry in rigor across datasets, driven by hardware/time constraints, not a methodological choice — must be stated plainly in the report's limitations.

**Key findings from this step:**
1. **RF's "complete collapse" on LendingClub and Home Credit was partly a default-hyperparameter artifact, not purely structural.** Proper tuning (specifically `min_samples_leaf`) recovers meaningful minority-class signal on both datasets (LendingClub F1: 0.000→0.118-0.147; Home Credit F1: 0.002→0.001-0.245 depending on resampling). This revises Early Finding #11's "unfixable" framing. **However, RF remains the weakest model on both datasets by a wide margin even after tuning** — this part of the original conclusion holds.
2. **XGBoost no-resampling remains the single best configuration on LendingClub** (F1=0.403±0.116, MCC=0.486±0.100) — confirms the original top-line finding survives tuning and full CV.
3. **SMOTE-ENN is the clear best resampling choice on Home Credit for every model** (F1 0.242-0.245 vs SMOTE's 0.190-0.209), and recovers much of the AUC-ROC that plain SMOTE sacrificed (0.705-0.711 vs 0.674-0.680) — a clean, strong, tightly-confirmed result.
4. **German Credit's "stacking wins" framing needs softening**: under tuning, XGBoost alone is competitive with or ahead of the stack in several conditions (e.g., no-resample: XGBoost F1=0.579 vs Stack F1=0.500).

Logged as Early Finding #12 (LendingClub) and #13 (all three datasets, complete). Full detail in `Early_Findings_Log.md`. **`Findings_Section.md` and `Discussion_Section_FINAL.md` both require substantial revision to replace single-split numbers and the old RF narrative with this tuned/CV-based picture.**

## Step 8 — SMOTE-ENN Comparison (closing the approved plan's Risk 3 gap)

Notebook: `08_smoteenn_comparison.ipynb`. Tested `imblearn.combine.SMOTEENN` against existing no-SMOTE/plain-SMOTE results, RF + XGBoost, all three datasets.

| Dataset | Model | No resampling (F1) | SMOTE (F1) | SMOTE-ENN (F1) |
|---|---|---|---|---|
| German Credit | RF | 0.525 | 0.561 | **0.635** |
| German Credit | XGBoost | 0.463 | 0.584 | 0.589 |
| Home Credit | RF | 0.002 | 0.016 | **0.182** |
| Home Credit | XGBoost | 0.059 | 0.062 | **0.235** |
| LendingClub | RF | 0.286 | 0.000 | 0.000 |
| LendingClub | XGBoost | 0.468 | 0.440 | **0.549** |

**Key finding:** SMOTE-ENN outperforms plain SMOTE on nearly every model/dataset combination — dramatically so on Home Credit (RF F1 0.002→0.182, XGBoost F1 0.059→0.235). The sole exception is RF on LendingClub, which still collapses to F1/G-mean=0.000 (MCC slightly negative) even with SMOTE-ENN's cleaning step. **This reframes RF-on-LendingClub as the true outlier in the study — not evidence that resampling generally hurts LendingClub, but that RF specifically cannot benefit from any synthetic oversampling once the minority class is as sparse as 142 training cases.** XGBoost benefits from resampling on all three datasets, most when combined with cleaning (SMOTE-ENN).

Logged as Early Finding #10 (updated after all 3 datasets tested — status: confirmed, single split, genuinely strengthens the RQ3 picture). Full detail in `Early_Findings_Log.md`. `Findings_Section.md` updated with a new "SMOTE-ENN as an alternative resampling strategy" subsection.

**Not yet done:** CV confirmation of SMOTE-ENN results; SMOTE-ENN on the stacked ensemble; SHAP stability check comparing SMOTE-ENN vs no-resampling (RQ4 analysis so far only compares plain SMOTE vs no resampling).

## Step 9 — Hyperparameter Investigation of RF's LendingClub Collapse

Notebook: `09_rf_tuning_lendingclub.ipynb`. Purpose: determine whether RF's collapse (Steps 4, 7, 8) is a fixable default-hyperparameter artifact or a genuine structural limitation.

Four configurations tested against the unadjusted baseline (F1=0.286, the best RF result on this dataset):
| Config | F1 | AUC-ROC |
|---|---|---|
| `class_weight='balanced'`, no resampling | 0.105 | 0.827 |
| Constrained depth (`max_depth=5, min_samples_leaf=10`), no resampling | 0.000 | 0.793 |
| Constrained depth + SMOTE-ENN | 0.077 | 0.701 |

**Result: every intervention tested made RF worse than doing nothing.** This rules out "fixable via tuning" — settled as a genuine structural limitation of RF's bagging mechanism on this severely sparse minority class (142 training cases), not a hyperparameter or resampling-technique problem. Practical conclusion: don't apply imbalance correction to RF when the minority class is this sparse in absolute terms; use XGBoost or the stacked ensemble instead (both benefit from resampling on the same data).

Logged as Early Finding #11 (status: confirmed, single split — considered a settled question for this project phase, no further tuning needed). Full detail in `Early_Findings_Log.md`. `Findings_Section.md` updated with a new "Ruling out a fixable hyperparameter explanation" subsection.

**Update — Home Credit SHAP stability CV check completed:** 2 of 5 folds run (memory-constrained: 300-row validation sample per fold, same sequential fit/explain/delete/gc.collect() pattern as the single-split version). Results: 0.926, 0.915 — closely matching the single-split value (0.906) and landing exactly between LendingClub's fold range (0.767-0.773) and German Credit's fold range (0.966-0.984), with zero overlap anywhere.

**The three-dataset SHAP stability ordering is now fully CV-confirmed at every level:**
| Dataset | CV fold range | Single-split value |
|---|---|---|
| German Credit | 0.966 – 0.984 | 0.983 |
| Home Credit | 0.915 – 0.926 | 0.906 |
| LendingClub | 0.767 – 0.773 | 0.797 |

This is now the most thoroughly validated result in the study — `Findings_Section.md` and `Early_Findings_Log.md` (entry #9) both updated accordingly.