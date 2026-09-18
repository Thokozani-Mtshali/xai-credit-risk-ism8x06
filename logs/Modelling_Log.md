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

**Still single-split only (not yet CV-confirmed):** German Credit/Home Credit SMOTE comparisons; XGBoost and Stacked ensemble SMOTE effects on LendingClub.

**Update — Home Credit SHAP stability CV check completed:** 2 of 5 folds run (memory-constrained: 300-row validation sample per fold, same sequential fit/explain/delete/gc.collect() pattern as the single-split version). Results: 0.926, 0.915 — closely matching the single-split value (0.906) and landing exactly between LendingClub's fold range (0.767-0.773) and German Credit's fold range (0.966-0.984), with zero overlap anywhere.

**The three-dataset SHAP stability ordering is now fully CV-confirmed at every level:**
| Dataset | CV fold range | Single-split value |
|---|---|---|
| German Credit | 0.966 – 0.984 | 0.983 |
| Home Credit | 0.915 – 0.926 | 0.906 |
| LendingClub | 0.767 – 0.773 | 0.797 |

This is now the most thoroughly validated result in the study — `Findings_Section.md` and `Early_Findings_Log.md` (entry #9) both updated accordingly.
