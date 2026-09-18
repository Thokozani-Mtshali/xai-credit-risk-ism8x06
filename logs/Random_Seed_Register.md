# Random Seed Register
*As committed to in Section 7.1 of the approved Data Collection Management and Quality Assurance Plan.*

## Seed value
**`RANDOM_SEED = 42`**, used consistently across every stage of this project, with no exceptions.

## Where it is applied
| Stage | Function/parameter | Notebook |
|---|---|---|
| Train/test split | `train_test_split(..., random_state=42)` | `03_split_and_cv.ipynb` |
| Cross-validation fold generation | `StratifiedKFold(..., random_state=42)` | `03_split_and_cv.ipynb` |
| Random Forest | `RandomForestClassifier(random_state=42, ...)` | `05_ensemble_models.ipynb`, `07_cv_confirmation.ipynb` |
| XGBoost | `XGBClassifier(random_state=42, ...)` | `04_baseline_logistic_regression.ipynb` (LogisticRegression itself does not require a seed for its default solver, but the surrounding pipeline does), `05_ensemble_models.ipynb`, `06_shap_analysis.ipynb`, `07_cv_confirmation.ipynb` |
| Stacked ensemble | `StackingClassifier(..., cv=StratifiedKFold(..., random_state=42))`, base/meta estimators each individually seeded | `05_ensemble_models.ipynb` |
| SMOTE | `SMOTE(random_state=42)` | `05_ensemble_models.ipynb`, `06_shap_analysis.ipynb`, `07_cv_confirmation.ipynb` |
| Logistic Regression (baseline and stacking meta-learner) | `LogisticRegression(..., random_state=42)` where applicable | `04_baseline_logistic_regression.ipynb`, `05_ensemble_models.ipynb` |
| SHAP evaluation sampling (Home Credit only, due to memory constraints) | `X.sample(n=..., random_state=42)` | `06_shap_analysis.ipynb`, `07_cv_confirmation.ipynb` |

## Rationale
A single fixed seed, reused everywhere rather than varied by stage or dataset, was chosen deliberately so that every reported result in this project is exactly reproducible from the saved code and data, and so that comparisons between models, datasets, and SMOTE conditions are not confounded by different random states producing incidental differences unrelated to the actual variable being tested.

## Note on cross-validation fold reuse
The 5-fold `StratifiedKFold` splits generated once in `03_split_and_cv.ipynb` (seeded with 42) were saved to `logs/fold_assignments/{dataset}_folds.json` and reused directly in `07_cv_confirmation.ipynb`, rather than being regenerated. This guarantees the CV-confirmation results in Step 7 are evaluated on the exact same fold partitions that were defined during data preparation, not a newly-seeded set that happens to share the same seed value.
