# Dataset Version and Checksum Record
*As committed to in Section 7.1 of the approved Data Collection Management and Quality Assurance Plan: "dataset version and checksum records, to confirm the data used matches what was originally downloaded."*

These SHA-256 checksums were generated locally via PowerShell's `Get-FileHash` on the raw source files, before any preprocessing was applied. Anyone re-running this project's notebooks can regenerate these checksums on their own downloaded copy and compare against the values below to confirm they are working from byte-identical source data.

| Dataset | File | SHA-256 checksum |
|---|---|---|
| German Credit | `german_credit_uci_original.csv` (retrieved via `ucimlrepo`, dataset ID 144) | `8A0D71DC2797E3488B15D3C810894E213ECF39A75B2620E6F32ABA439258582E` |
| Home Credit Default Risk | `application_train.csv` (Kaggle competition download) | `52E96B895B1112E1C853F670E58372719C8441C5ED1C57AC2F7FAD559D784F5F` |
| LendingClub | `loans_full_schema.csv` | `52872E7D4F8A388B7B4891852EA9009FBA002DF63AD53D8DBC8EA97B982B2E29` |

## Regenerating these checksums
```powershell
Get-FileHash "Data\Raw\German Credit Risk Dataset\german_credit_uci_original.csv" -Algorithm SHA256
Get-FileHash "Data\Raw\Home Credit Default Risk Dataset\application_train.csv" -Algorithm SHA256
Get-FileHash "Data\Raw\Lending Club Loan Dataset\loans_full_schema.csv" -Algorithm SHA256
```

## Notes
- These checksums were generated after the German Credit dataset source correction (see `Deviations_From_Approved_Plan.md`, item 1) — they reflect the genuine UCI 20-attribute dataset actually used throughout the project, not the initially-downloaded incorrect 10-column Kaggle file.
- Checksums were not generated for the processed/cleaned versions of each dataset (e.g. `application_train_encoded.csv`), since those are fully reproducible from the raw files via the documented notebooks (`01_data_diagnosis.ipynb` through `03_split_and_cv.ipynb`) and are excluded from version control via `.gitignore` in any case.