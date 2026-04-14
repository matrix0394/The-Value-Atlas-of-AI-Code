## `data/` Directory

This directory contains the processed analysis inputs expected by the cleaned
GitHub code repository.

### Subdirectories

- `country_values/`
  Fixed benchmark coordinates and the frozen PCA model used for projection.
- `external/`
  Country-level regression covariates used to regenerate the publication-ready
  regression table.
- `llm_interviews/intrinsic/`
  Released processed intrinsic interview tables.
- `llm_interviews/multilingual/`
  Released processed multilingual roleplay interview tables.
- `llm_pca/`
  Precomputed intrinsic and roleplay coordinate tables.
- `raw/`
  Placeholder only; the raw IVS `.sav` file is not included in this public
  package.

### File formats

- `.csv`
  Tabular analysis inputs.
- `.json`
  Structured metadata and processed tables.
- `.pkl`
  Serialized Python objects used by the accompanying analysis code.

Pickle files can be sensitive to library versions, especially across different
`pandas` and `scikit-learn` releases. They are most reliably used with the
accompanying code repository and its documented Python environment.

### Notes

When benchmark reconstruction from raw survey data is needed, the IVS `.sav`
input should be recreated from the official EVS Trend File 1981-2017 (Version
`3.0.0`; DOI `10.4232/1.14021`) and the WVS Trend File 1981-2022 (Version
`4.1.0`; DOI `10.14281/18241.27`).

This public archive does not include the full `interview_raw/` API-response
caches. Those caches were used during private data collection, but they are not
required to reproduce the paper results from the released processed tables and
projected coordinates.
