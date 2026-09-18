# Deviations from the Approved Plan
*As committed to in Section 7.1 of the approved Data Collection Management and Quality Assurance Plan: "a record of any deviations from the originally approved plan, with justification." Deviations are drawn together here from across the project's working logs (`Preprocessing_Decision_Log.md`, `Modelling_Log.md`, `Early_Findings_Log.md`) for a single point of reference.*

## 1. German Credit dataset source correction
**Deviation:** the initially downloaded file (`german_credit_data.csv`, from a Kaggle upload) was a simplified 10-column derivative, not the 20-attribute dataset named in the approved proposal.
**Justification:** this was identified and corrected before any analysis was performed on the incorrect file, by retrieving the genuine UCI Statlog German Credit dataset via the `ucimlrepo` package. The dataset named in the proposal (German Credit, 20 attributes) is the one actually used throughout; no substitution of the *intended* dataset occurred, only a correction of an initial sourcing error.

## 2. Exceptions to the 20%-missingness drop threshold (Home Credit)
**Deviation:** the approved plan's stated rule — "any feature with more than 20% missing values will be dropped and the decision logged" — was not applied uniformly. `OWN_CAR_AGE` (66.0% missing) and `OCCUPATION_TYPE` (31.3% missing) were retained rather than dropped.
**Justification:** cross-tabulation evidence showed both variables' missingness was Missing At Random and structurally explainable (car age missing almost perfectly corresponds to non-car-ownership; occupation missingness concentrated among pensioners), rather than reflecting genuine data unavailability. Both variables were judged plausibly relevant to credit risk. This is consistent with the plan's own stated intent that the missingness pattern be diagnosed first "rather than imputed blindly" (Section 5.2); the 20% figure is applied here as a default requiring justified exception, not an inflexible rule.

## 3. LendingClub target variable construction
**Deviation:** the approved proposal does not specify how LendingClub's `loan_status` field should be converted into a binary default/non-default target. Only 7 of 10,000 rows had reached a fully resolved "Charged Off" status at the time of data extraction.
**Justification:** restricting analysis to fully resolved loans (Fully Paid vs Charged Off only) would have left 454 usable rows with just 7 positive cases — not viable for model training or evaluation. A broader definition (any indication of repayment difficulty treated as the positive class) was adopted instead, following the majority convention identified across multiple comparable published studies. This is stated explicitly as a limitation in the Findings Section, since "Current" loans are treated as non-default despite their eventual outcome being unknown.

## 4. SMOTE-ENN not yet tested
**Deviation:** the approved plan's Risk 3 mitigation specifies "SMOTE and SMOTE-ENN applied only to training folds." Only plain SMOTE has been tested to date; SMOTE-ENN has not yet been run on any dataset.
**Justification:** none yet — this is an acknowledged gap against the approved plan, not a considered substitution, and is intended to be addressed in the next phase of the project.

## 5. Hyperparameters left at default values
**Deviation:** the approved plan describes "hyperparameter search scoped to a reduced but representative grid" (Risk 4 mitigation). All models to date (Random Forest, XGBoost, the stacked ensemble) have used scikit-learn/XGBoost default hyperparameters, with no grid search performed.
**Justification:** default hyperparameters were used deliberately at this stage to establish a fair, consistent comparative baseline across three datasets, three model types, and two SMOTE conditions, before committing computational resources to tuning any single configuration. This is stated as a limitation in the Findings Section. A scoped tuning pass is planned but not yet completed.

## 6. Home Credit SHAP evaluation sample size
**Deviation:** the approved plan's Risk 4 mitigation anticipated computing SHAP "on stratified samples where full-dataset computation is infeasible," without specifying a sample size. In practice, computation required a smaller sample (300 rows) than initially attempted (1,000 rows), due to memory constraints on the available local hardware.
**Justification:** this is a direct, anticipated instance of the risk the plan already flagged, resolved via the mitigation the plan already proposed (sampling), simply requiring a smaller sample than first attempted. Full detail in `SHAP_Configuration_Log.md`.

## 7. Outstanding audit trail items not yet completed
The following items, promised in Section 7.1 of the approved plan, have not yet been created:
- Git/GitHub version-controlled repository with commit history
- Dataset version/checksum records
- A single consolidated data dictionary/codebook cross-referencing all three datasets' features (currently scattered across `Preprocessing_Decision_Log.md` and inline code comments)

These are acknowledged gaps, not deliberate substitutions, and are intended to be addressed as the project continues.
