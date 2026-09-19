# Data Dictionary and Codebook
*As committed to in Section 7.1 of the approved Data Collection Management and Quality Assurance Plan: "a data dictionary and codebook log cross-referencing each feature against its original documentation."*

This consolidates feature-level documentation scattered across `Preprocessing_Decision_Log.md`, `Modelling_Log.md`, and inline notebook comments into a single reference. For each dataset: original meaning, any encoding/transformation applied, and current status in the modelling-ready data.

---

## 1. German Credit Dataset (UCI Statlog, `german_credit_uci_original.csv`)

Source: UCI Machine Learning Repository, dataset ID 144, retrieved via `ucimlrepo`. 1,000 observations, 20 original attributes + target.

| Feature (final name) | Original UCI attribute | Meaning | Type | Transformation applied |
|---|---|---|---|---|
| `status_checking_account` | Attribute1 | Status of existing checking account (4 categories, e.g. "< 0 DM") | Categorical | Decoded from A11-A14 codes; one-hot encoded |
| `duration_months` | Attribute2 | Loan duration in months | Numeric | IQR outlier-capped (0 outliers found); min-max scaled |
| `credit_history` | Attribute3 | Credit history (5 categories) | Categorical | Decoded from A30-A34 codes; one-hot encoded |
| `purpose` | Attribute4 | Loan purpose (11 categories, e.g. car, education) | Categorical | Decoded from A40-A410 codes; one-hot encoded |
| `credit_amount` | Attribute5 | Loan amount requested | Numeric | IQR outlier-capped (0 outliers found); min-max scaled |
| `savings_account` | Attribute6 | Savings account/bonds level (5 categories, ordinal) | Ordinal | Decoded from A61-A65 codes; ordinal-encoded 0-4 |
| `employment_since` | Attribute7 | Length of current employment (5 categories, ordinal) | Ordinal | Decoded from A71-A75 codes; ordinal-encoded 0-4 |
| `installment_rate_pct` | Attribute8 | Installment rate as % of disposable income | Numeric | Min-max scaled |
| `personal_status_sex` | Attribute9 | Combined marital status and sex (4 categories) | Categorical | Decoded from A91-A95 codes; one-hot encoded |
| `other_debtors_guarantors` | Attribute10 | Other debtors/guarantors (3 categories) | Categorical | Decoded from A101-A103 codes; one-hot encoded |
| `residence_since` | Attribute11 | Years at current residence | Numeric | Min-max scaled |
| `property` | Attribute12 | Property type (4 categories) | Categorical | Decoded from A121-A124 codes; one-hot encoded |
| `age` | Attribute13 | Applicant age in years | Numeric | IQR outlier-capped (0 outliers found); min-max scaled |
| `other_installment_plans` | Attribute14 | Other installment plans (3 categories) | Categorical | Decoded from A141-A143 codes; one-hot encoded |
| `housing` | Attribute15 | Housing type (3 categories) | Categorical | Decoded from A151-A153 codes; one-hot encoded |
| `existing_credits_count` | Attribute16 | Number of existing credits at this bank | Numeric | Min-max scaled |
| `job` | Attribute17 | Job/skill level (4 categories, ordinal) | Ordinal | Decoded from A171-A174 codes; ordinal-encoded 0-3 |
| `num_dependents` | Attribute18 | Number of people financially dependent on applicant | Numeric | Min-max scaled |
| `telephone` | Attribute19 | Telephone registered (binary) | Categorical | Decoded from A191-A192; one-hot encoded |
| `foreign_worker` | Attribute20 | Foreign worker status (binary) | Categorical | Decoded from A201-A202; one-hot encoded |
| `target` | class | 1=Good credit, 2=Bad credit (original coding) | Binary | Recoded to 0=Good, 1=Bad (default) for modelling |

**Missingness:** none, confirmed via UCI documentation and independent verification.
**Final modelling shape:** 1,000 rows × 41 columns (post one-hot expansion).

---

## 2. Home Credit Default Risk Dataset (`application_train.csv` only)

Source: Kaggle competition dataset. 307,511 raw rows, 122 raw columns. Only the main application table was used; supplementary linked tables (`bureau`, `bureau_balance`, `credit_card_balance`, `installments_payments`, `POS_CASH_balance`, `previous_application`) were deliberately excluded from this phase (see `Deviations_From_Approved_Plan.md`, item 7, for related outstanding items). `application_test.csv` and `sample_submission.csv` (Kaggle-competition artifacts with no target column) were not used at all.

### Columns dropped entirely (>20% missing, no viable recovery)
48 columns, almost all building/property physical-characteristic fields: `COMMONAREA_AVG/MODE/MEDI`, `NONLIVINGAPARTMENTS_AVG/MODE/MEDI`, `FONDKAPREMONT_MODE`, `LIVINGAPARTMENTS_AVG/MODE/MEDI`, `FLOORSMIN_AVG/MODE/MEDI`, `YEARS_BUILD_AVG/MODE/MEDI`, `LANDAREA_AVG/MODE/MEDI`, `BASEMENTAREA_AVG/MODE/MEDI`, `NONLIVINGAREA_AVG/MODE/MEDI`, `ELEVATORS_AVG/MODE/MEDI`, `WALLSMATERIAL_MODE`, `APARTMENTS_AVG/MODE/MEDI`, `ENTRANCES_AVG/MODE/MEDI`, `LIVINGAREA_AVG/MODE/MEDI`, `HOUSETYPE_MODE`, `FLOORSMAX_AVG/MODE/MEDI`, `YEARS_BEGINEXPLUATATION_AVG/MODE/MEDI`, `TOTALAREA_MODE`, `EMERGENCYSTATE_MODE` — plus `EXT_SOURCE_1` (external bureau score, 56.4% missing, likely genuine data unavailability rather than structural non-applicability).

### Key retained/engineered columns
| Feature | Meaning | Missingness handling | Transformation |
|---|---|---|---|
| `TARGET` → `target` | 1=default, 0=no default | None (target variable) | Column renamed to lowercase for consistency with other datasets |
| `SK_ID_CURR` | Unique application ID | None | Dropped before modelling (identifier, not a feature) |
| `CODE_GENDER` | Applicant gender (F/M/XNA) | 4 rows with placeholder "XNA" dropped entirely | One-hot encoded (F/M only, post-drop) |
| `OWN_CAR_AGE` | Age of applicant's car | 66.0% missing — **retained despite exceeding threshold**; confirmed MAR via cross-tab against `FLAG_OWN_CAR` (202,924/202,929 missing = non-owners) | Imputed 0 for non-owners; median-among-owners for 5 genuine gaps |
| `OCCUPATION_TYPE` | Applicant occupation category (18 categories) | 31.3% missing — **retained**; MAR-ish (concentrated among pensioners) but messier than `OWN_CAR_AGE` | Imputed as new category "Unknown/Not Applicable"; frequency-encoded (58-category `ORGANIZATION_TYPE` handled the same way, not individually listed here) |
| `DAYS_EMPLOYED` | Days employed before application (negative = past) | Contained placeholder value 365243 for 55,374 rows (18.0%), corresponding to retired/unemployed applicants | Flag column `DAYS_EMPLOYED_ANOMALY` created (1 if placeholder); placeholder replaced with 0 |
| `AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK/MON/QRT/YEAR` | Number of credit bureau inquiries in various time windows | 13.5% missing each, confirmed missing together as a block (41,519 rows) = "no inquiry history" | Imputed 0 |
| `EXT_SOURCE_2`, `EXT_SOURCE_3` | External normalized credit bureau scores | 0.2% / 19.8% missing | Median imputation |
| `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` | Income, credit amount, annuity, goods price | Minor missingness on some (median-imputed) | IQR outlier-capped (floored at 0); `AMT_INCOME_TOTAL` had 14,035 outliers capped (known extreme high-income cases in this dataset, original max 117,000,000) |
| `NAME_TYPE_SUITE`, `CNT_FAM_MEMBERS`, `OBS/DEF_30/60_CNT_SOCIAL_CIRCLE`, `DAYS_LAST_PHONE_CHANGE` | Various | All <1% missing | Mode (categorical) / median (numeric) imputation |
| `NAME_CONTRACT_TYPE`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY` | Binary flags | None | Mapped directly to 0/1 |
| `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `WEEKDAY_APPR_PROCESS_START` | Low-cardinality categoricals (5-8 categories each) | None significant | One-hot encoded |

**Final modelling shape:** 307,507 rows (after XNA drop) × 74 columns pre-one-hot, expanding to 101-102 columns post-encoding.

---

## 3. LendingClub Loan Dataset (`loans_full_schema.csv`)

Source: publicly distributed community dataset (originally compiled by Kevin Kuo, commonly hosted on Kaggle). 10,000 rows, ~55 raw variables. Verified as a legitimate standalone dataset, not a trimmed derivative of a larger "original" (unlike the initial German Credit sourcing issue).

### Target construction (not a raw column — engineered)
| Feature | Construction |
|---|---|
| `target` | Built from `loan_status` (6 raw categories: Current, Fully Paid, In Grace Period, Late (16-30 days), Late (31-120 days), Charged Off). 1 (default/risk) = Charged Off + both Late categories + In Grace Period; 0 (non-default) = Fully Paid + Current. Chosen over a stricter "fully resolved only" definition, which would have left only 454 usable rows (see `Deviations_From_Approved_Plan.md`, item 3). |

### Key retained/engineered columns
| Feature | Meaning | Missingness handling | Transformation |
|---|---|---|---|
| `emp_title` | Free-text job title (4,742 unique values) | 8.3% missing | **Dropped entirely** — too high-cardinality/unstructured to encode meaningfully |
| `emp_length` | Years employed | 8.2% missing | Imputed 0 (plausible for unemployed/undisclosed) |
| `annual_income_joint`, `debt_to_income_joint`, `verification_income_joint` | Joint-application income fields | ~85% missing each; confirmed 100% structural MAR via cross-tab against `application_type` (missing exactly when application is individual, not joint) | Imputed 0 (numeric) / "Not Applicable" (categorical) |
| `months_since_last_delinq`, `months_since_90d_late`, `months_since_last_credit_inquiry` | Time since last delinquency/inquiry event | 56.6% / 77.2% / 12.7% missing; confirmed MAR via cross-tab against `delinq_2y` (99.8% of missing cases had zero delinquencies — "event never occurred," not unknown) | Imputed with sentinel value 999 ("no event on record") — noted as a modelling convenience suited to tree-based models |
| `num_accounts_120d_past_due`, `debt_to_income` | Various | 3.2% / 0.24% missing | Median imputation |
| `grade` | LendingClub credit grade (A-G) | None | Ordinal-encoded 0-6 (A=0/best to G=6/worst) |
| `sub_grade` | Sub-grade within grade (A1-G5, 32 categories) | None | Ordinal-encoded 0-31, built programmatically to preserve natural ranking |
| `state` | US state (50 categories) | None | Frequency-encoded (too high-cardinality for one-hot) |
| `homeownership`, `verified_income`, `loan_purpose`, `application_type`, `issue_month`, `initial_listing_status`, `disbursement_method` | Various low-cardinality categoricals | None significant | One-hot encoded |
| `annual_income`, `loan_amount`, `interest_rate`, `installment`, `balance`, `paid_total`, `paid_principal`, `paid_interest`, `total_credit_limit`, `total_credit_utilized` | Continuous financial variables | None significant | IQR outlier-capped (floored at 0); `debt_to_income` also had 33 rows (0.33%) with implausible values (max 469.09), capped via the same standard IQR rule |

**Final modelling shape:** 10,000 rows × 70 columns (post one-hot expansion).

---

## Cross-dataset notes

- **Target column naming standardized** to lowercase `target` across all three datasets during Step 2 (Home Credit's raw column was `TARGET`).
- **Column name sanitization:** a `clean_column_names()` function (regex-replacing `[`, `]`, `<`, `>` with `_`) was applied to all three datasets before any XGBoost use, since XGBoost rejects feature names containing those characters (several one-hot-encoded category labels, e.g. German Credit's "< 0 DM", contained them).
- **Scaling:** all continuous features were min-max scaled to [0,1] using `MinMaxScaler` fit on training data only (per dataset), applied after the train/test split to avoid data leakage.
- **Encoding strategy summary:** ordinal encoding used only where a genuine natural ordering exists (German Credit's `savings_account`/`employment_since`/`job`; LendingClub's `grade`/`sub_grade`); one-hot encoding for low-cardinality nominal categories; frequency encoding for high-cardinality nominal categories (Home Credit's `ORGANIZATION_TYPE`/`OCCUPATION_TYPE`, LendingClub's `state`) to avoid excessive sparse columns.