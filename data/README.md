This directory is intentionally kept lightweight in the GitHub repository.

The processed datasets needed by the code are expected to be supplied from the
Dryad archive and placed into the same relative paths shown below:

- `country_values/`
  Fixed IVS benchmark coordinates and the saved PCA model.
- `external/`
  Upstream regression covariates (`regression_covariates.csv`).
- `llm_interviews/intrinsic/`
  Cached intrinsic multilingual LLM answers and derived IVS-format tables.
- `llm_interviews/multilingual/`
  Cached multilingual country-roleplay answers and derived IVS-format tables.
- `llm_pca/`
  Precomputed intrinsic and roleplay PCA coordinate tables.
- `raw/`
  Optional compatibility location for the raw IVS `.sav` file.

In the original working project, the raw IVS file was commonly stored under:

- `country_values/Integrated_values_surveys_1981-2022.sav`

The cleaned repository accepts both:

- `data/country_values/Integrated_values_surveys_1981-2022.sav`
- `data/raw/Integrated_values_surveys_1981-2022.sav`
