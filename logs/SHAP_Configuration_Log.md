# SHAP Configuration Log
*As committed to in Section 7.1 of the approved Data Collection Management and Quality Assurance Plan: "a SHAP configuration log recording the explainer type and any background or reference sample used."*

## Explainer type
`shap.TreeExplainer`, used throughout for all three datasets. Chosen because it is the exact-computation explainer designed for tree-based ensemble models, and is applicable directly to XGBoost without approximation.

## Background/reference sample
No custom background dataset was supplied to `TreeExplainer` (the `data` parameter was left at its default of `None`). For tree-based models, `TreeExplainer`'s default behaviour uses the model's own internal tree structure (conditional expectations derived from the trees themselves) rather than requiring an external background sample, which is the standard and recommended approach for this explainer type and distinguishes it from explainers such as `KernelExplainer` that do require an explicit background set.

## Model explained
In every case, the model passed to `TreeExplainer` was a fitted **XGBoost classifier**, not the stacked ensemble or Random Forest. This was a deliberate choice: XGBoost performed close to the stacked ensemble across all three datasets (Section 3 of the Findings Section) while being directly and unambiguously explainable by `TreeExplainer`, whereas explaining a `StackingClassifier`'s combined output is a less standard and more complex undertaking, not attempted in this phase of the project.

## Evaluation sample size per dataset
| Dataset | Evaluation rows used | Reason |
|---|---|---|
| German Credit | Full test set (200 rows) | Small enough to explain in full without memory concern |
| LendingClub | Full test set (2,000 rows) | Small enough to explain in full without memory concern |
| Home Credit — single-split analysis | 1,000-row random sample (`random_state=42`) initially attempted; reduced to 300-row random sample after continued memory failure | Full test set (61,502 rows) triggered `MemoryError` when computing `TreeExplainer.shap_values()` alongside a second fitted model in memory; resolved via a sequential fit-explain-delete pattern (see below) combined with sample reduction |
| Home Credit — cross-validation check | 300-row random sample per fold (`random_state=42`) | Same memory constraint as above, applied per fold rather than to a single held-out test set |

## Memory-management approach (Home Credit specifically)
Fitting both the SMOTE and non-SMOTE XGBoost models before computing SHAP values for either caused repeated `MemoryError` failures on Home Credit's 246,005-row, 100-feature training set. This was resolved by processing one model at a time: fit the first model, compute its SHAP values and reduce them immediately to a feature-importance ranking (`abs(shap_values).mean(axis=0)`), then explicitly delete the fitted model, explainer, and raw SHAP value array (`del` statement) and invoke `gc.collect()` before fitting the second model. This pattern was applied consistently for both the single-split Home Credit analysis and the subsequent cross-validation check, and is the approach that should be reused for any future SHAP computation on Home Credit at full or near-full scale.

## Metric used for stability comparison
Feature importance was summarised as the **mean absolute SHAP value** per feature across the evaluation sample (`abs(shap_values).mean(axis=0)`), ranked in descending order. Stability between the SMOTE and non-SMOTE conditions was then quantified using **Spearman rank correlation** between the two resulting rankings, as committed to in the approved proposal's Risk 6 mitigation ("computing Spearman rank correlation of SHAP rankings across folds and across SMOTE/no-SMOTE conditions").
