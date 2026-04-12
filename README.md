# The Value Atlas of AI

This repository accompanies the manuscript *The Value Atlas of AI: Mapping
World Human Values in Large Language Models*. In the paper, we study how 20
large language models represent cultural values across 66 countries and
territories by projecting model responses into the Inglehart-Welzel value
space and comparing them with human benchmark coordinates derived from the
Integrated Values Survey. The analyses reported in the manuscript identify a
strong secular-rational baseline bias, pronounced prompt-language effects, an
asymmetric English advantage in the representation of many non-Western
societies, and associations between these patterns and historical colonial
structures.

This repository provides the code used to reproduce those reported analyses.
It is designed to work together with the Dryad deposit for the paper, which
contains the released processed datasets, projected coordinates, and
publication-ready source tables. The public release is split deliberately:

- `GitHub`: code, configuration files, and reproducibility instructions
- `Dryad`: processed data, supplementary source tables, and figure source data

The directory layout follows the working project as closely as possible, so the
Dryad package can be copied into the repository root without rewriting file
paths.

## Paper Overview

At a high level, the project combines three pieces:

- a fixed human benchmark built from IVS/WVS cultural-value coordinates
- multilingual LLM response collection and recoding using ten WVS-derived items
- downstream distance-based analyses of language effects, regional asymmetries,
  and historical covariates

## Quick Start

For most readers and reviewers, the recommended path is to reproduce the paper
from the released processed data rather than rebuild the full pipeline from raw
survey files or private API-response caches. The shortest public workflow is:

1. Download the Dryad package and extract it.
2. Copy the unpacked `data/`, `results/`, and `Supplementary Materials/`
   directories into the root of this repository so that they merge with the
   placeholder directories already present here.
3. Install the Python dependencies listed in `requirements.txt` (or
   `requirements.lock.txt` if you want the closest match to the tested
   environment).
4. Run `python src/run/run_paper_analysis.py` to regenerate the paper-level
   analysis tables.
5. Run the figure scripts in `src/figures/` only if you want to regenerate the
   released main-text or supplementary figures.

This path does not require rerunning private API calls or rebuilding the IVS
benchmark from the raw survey file.

## Workflow Overview

The public release is organized around a processed-data-first workflow. The
diagram below shows how the released Dryad package, the optional raw inputs,
and the main analysis scripts fit together.

```mermaid
flowchart TD
    A["Dryad package"] --> B["Processed benchmark files<br/>data/country_values/"]
    A --> C["Processed interview tables<br/>data/llm_interviews/"]
    A --> D["Projected coordinate tables<br/>data/llm_pca/"]
    A --> E["Publication-ready source data<br/>Supplementary Materials/data/"]

    F["Optional raw IVS file<br/>data/raw/*.sav"] --> G["run_country_values_analysis.py"]
    G --> B

    H["Optional private API reruns"] --> I["run_llm_values_analysis.py"]
    H --> J["run_llm_multilingual_analysis.py"]
    H --> K["run_roleplay_multilingual_analysis.py"]

    I --> C
    J --> C
    K --> C

    C --> L["Response validation and IVS recoding<br/>base_interview.py + ivs_question_processor.py"]
    L --> D
    B --> M["run_paper_analysis.py"]
    D --> M
    A --> N["Regression covariates<br/>data/external/regression_covariates.csv"]
    N --> M
    E --> O["Figure scripts<br/>src/figures/*.py"]
    M --> P["Paper outputs<br/>results/analysis/ + results/paper_data/"]
    P --> O
    O --> Q["Main-text and supplementary figures"]
```

## What The Public Release Contains

This release is organized around the processed data used in the paper rather
than the full private data-collection workflow. The repository contains the
analysis code and configuration, while the Dryad package contains the released
processed interview tables, PCA coordinate files, regression inputs, and
publication-ready source data. The original IVS/WVS/EVS raw survey file is not
redistributed here.

The multilingual prompt templates and language-specific instructions are
preserved as released experimental materials in order to maintain fidelity to
the study design. Those strings are part of the core experimental stimuli, not
incidental comments.

## System Requirements

- Python `3.10` or later
- Standard scientific Python stack as listed in `requirements.txt`
- No GPU is required for the released processed-data reproduction path
- Private reruns of model calls require API credentials in `.env`
- Tested in the project development environment on macOS `15.6.1`
- The released processed-data workflow is intended to run on a standard
  workstation or laptop

Installation of the Python dependencies typically takes only a few minutes on a
standard machine. Runtime depends on whether you use the released processed
data or rerun private model calls.

For reproducibility, two dependency files are provided:

- `requirements.txt` lists the minimum supported package versions.
- `requirements.lock.txt` records the exact package versions used in the
  tested development environment.

If you need the closest possible match to the tested environment, install from
`requirements.lock.txt`. Using newer package versions may still work, but can
introduce small numerical differences in projected coordinates and figure
rendering.

## API Credentials

The public processed-data reproduction path does not require any API keys.
Private reruns of model calls do.

To configure API access:

1. Copy `.env.example` to `.env`.
2. Fill in the two key variables used by the released model registry:
   - `OPENKEY_API_KEY`
   - `OPENROUTER_API_KEY`

These variable names come directly from `config/models/llm_models.json`.
The released configuration uses OpenKey and OpenRouter for unified access
across providers. If you prefer official provider endpoints, you can edit the
model-specific `base_url` and credential fields in
`config/models/llm_models.json`.

## Cost And Runtime Notes

The released processed-data path does not incur API cost.

Private reruns of the full model-interview pipelines can be substantially more
time-consuming and expensive. A full rerun spans 20 models, 66 countries and
territories, 14 released prompt languages, and 10 IVS/WVS-derived items. The
exact monetary cost depends on provider pricing, enabled models, request
concurrency, and rate limits at the time of execution, so we have not included
a single stable USD estimate in this public README.

## Generation Settings And Stochasticity

The public reproduction path relies on the released processed interview tables
and projected coordinates, not on byte-identical API reruns.

In the released code, model calls preserve the study rerun setting
(`temperature = 0.1` in `src/base/base_interview.py`) to reduce stochastic
variation while matching the configuration used during the project. The code
does not currently set a provider-level `seed`, so exact API outputs may still
vary across providers and over time. Downstream PCA reconstruction uses a
fixed NumPy seed (`np.random.seed(42)` in `src/base/base_pca_analyzer.py`)
where randomized numerical routines are used.

## Expected Outputs

Running

```bash
python src/run/run_paper_analysis.py
```

produces the main paper-level outputs in:

- `results/analysis/`
- `results/paper_data/`

Key files include:

- `results/analysis/study1_intrinsic_bias.json`
- `results/analysis/study2_english_advantage.json`
- `results/analysis/study3_digital_orientalism.json`
- `results/analysis/study4_colonial_legacies.json`
- `results/analysis/model_imitation_accuracy.csv`
- `results/analysis/paper_statistics_all.json`
- `results/paper_data/regression_data.csv`

The figure scripts in `src/figures/` write rendered outputs to:

- `results/figures/paper/`
- `Supplementary Materials/figures/`

## Repository Layout

- `src/run/`
  Entry-point scripts for benchmark construction, intrinsic model analysis,
  multilingual intrinsic analysis, multilingual roleplay analysis, and final
  paper-level analysis.
- `src/base/`
  Shared utilities for model calls, response validation, recoding, and
  PCA-based projection.
- `src/country_values/`
  Human-benchmark construction from IVS data.
- `src/llm_values/`
  Intrinsic model interviewing, response processing, and coordinate
  computation.
- `src/roleplay_multilingual/`
  Multilingual country-roleplay interviewing, response processing, and
  coordinate computation.
- `src/figures/`
  Scripts used to regenerate the released main-text and supplementary figures.
- `config/`
  Model registry, questionnaire definitions, and country metadata.
- `data/`
  Expected location for released processed data and, optionally, the raw IVS
  `.sav` file.
- `results/`
  Output directories written by the analysis and figure scripts.
- `Supplementary Materials/data/`
  Publication-ready supplementary tables and figure source data from Dryad.

## Expected Dryad Layout

The released Dryad package is expected to mirror the current project
structure. The key locations are:

```text
data/
  country_values/
  external/
  llm_interviews/
    intrinsic/
    multilingual/
      processed/
  llm_pca/
    intrinsic/
    multilingual/
  raw/

results/
  paper_data/

Supplementary Materials/
  data/
```

The only path that may vary from the original working project is the raw IVS
survey file. The benchmark code accepts either:

- `data/country_values/Integrated_values_surveys_1981-2022.sav`
- `data/raw/Integrated_values_surveys_1981-2022.sav`

The raw benchmark source is not redistributed in this repository or in the
public Dryad package. In the original project, this `.sav` file was derived
from the Integrated Values Survey (IVS), constructed by merging:

- the EVS Trend File 1981-2017, Version `3.0.0` (DOI: `10.4232/1.14021`)
- the WVS Trend File 1981-2022, Version `4.1.0` (DOI: `10.14281/18241.27`)

If you want to rebuild the benchmark from raw survey data, please obtain those
official source files directly from the EVS/WVS providers and recreate the
merged IVS input used by the benchmark pipeline.

## Minimal Data Needed To Reproduce The Paper

To rerun the reported paper statistics from the released processed data, the
following files are sufficient:

```text
data/
  country_values/
    country_scores_pca.json
    pca_model_fixed.pkl
  external/
    regression_covariates.csv
  llm_pca/
    intrinsic/
      llm_pca_entity_scores.pkl
    multilingual/
      roleplay_ml_pca_entity_scores_latest.pkl

results/
  paper_data/
    regression_data.csv

Supplementary Materials/
  data/
    figure2_baseline_20models.csv
    model_imitation_accuracy.csv
    study5_model_imitation.json
    DataS1_ivs_pca_coordinates.csv
    ivs_pca_coordinates.csv
    llm_roleplay_pca.csv
    figure3_digital_orientalism.csv
    study3_digital_orientalism.json
    figure4_colonial_history.csv
```

`results/paper_data/regression_data.csv` is the released paper table used for
the manuscript source data. The separate file
`data/external/regression_covariates.csv` is an upstream lookup table that
supplies country-level covariates needed when regenerating the paper table from
the coordinate data.

## Reproduction Levels

### 1. Reproduce the paper results from released processed data

This is the recommended path for peer review.

Run:

```bash
python src/run/run_paper_analysis.py
```

Main inputs:

- `data/country_values/country_scores_pca.json`
- `data/country_values/pca_model_fixed.pkl`
- `data/llm_pca/intrinsic/llm_pca_entity_scores.pkl`
- `data/llm_pca/multilingual/roleplay_ml_pca_entity_scores_latest.pkl`
- `data/external/regression_covariates.csv`
- `results/paper_data/regression_data.csv`

Main outputs:

- `results/analysis/study1_intrinsic_bias.json`
- `results/analysis/study2_english_advantage.json`
- `results/analysis/study3_digital_orientalism.json`
- `results/analysis/study4_colonial_legacies.json`
- `results/analysis/model_imitation_accuracy.csv`
- `results/analysis/model_imitation_accuracy.json`
- `results/analysis/paper_statistics_all.json`
- `results/paper_data/`

### 2. Rebuild coordinate datasets from released processed interview tables

This level reconstructs the intrinsic and roleplay coordinate datasets from the
processed interview tables released in Dryad, without rerunning API calls.

Intrinsic interview tables:

- `data/llm_interviews/intrinsic/multilingual_ivs_format.pkl`
- `data/llm_interviews/intrinsic/multilingual_ivs_format.json`
- `data/llm_interviews/intrinsic/multilingual_processed_20260311_221147.pkl`
- `data/llm_interviews/intrinsic/multilingual_processed_20260311_221147.json`

Roleplay interview tables:

- `data/llm_interviews/multilingual/verified_data_for_report.json`
- `data/llm_interviews/multilingual/processed/llm_roleplay_ml_processed_responses_ivs_format_latest.pkl`
- `data/llm_interviews/multilingual/processed/multilingual_ivs_format.pkl`
- `data/llm_interviews/multilingual/processed/multilingual_ivs_format.json`
- `data/llm_interviews/multilingual/processed/roleplay_ml_language_analysis_latest.json`

### 3. Rebuild the human benchmark from raw IVS data

This level reconstructs the benchmark PCA model and country coordinates from
the original IVS `.sav` file. It is not required for reproducing the reported
paper results.

## Analytical Workflow

### Human benchmark

The human benchmark is built by:

```bash
python src/run/run_country_values_analysis.py
```

Core files:

- `src/run/run_country_values_analysis.py`
- `src/country_values/data_processing.py`
- `src/country_values/pca_analysis.py`
- `src/base/base_pca_analyzer.py`

Required raw input:

- `data/country_values/Integrated_values_surveys_1981-2022.sav`
  or `data/raw/Integrated_values_surveys_1981-2022.sav`

This input corresponds to the IVS benchmark derived from the EVS Trend File
1981-2017 (Version `3.0.0`; DOI `10.4232/1.14021`) and the WVS Trend File
1981-2022 (Version `4.1.0`; DOI `10.14281/18241.27`).

Outputs:

- `data/country_values/country_scores_pca.json`
- `data/country_values/pca_model_fixed.pkl`

This stage defines the fixed coordinate space used throughout the rest of the
project.

### Intrinsic model-value pipeline

The baseline intrinsic pipeline is controlled by:

```bash
python src/run/run_llm_values_analysis.py
```

Core files:

- `src/run/run_llm_values_analysis.py`
- `src/llm_values/llm_interview.py`
- `src/llm_values/llm_data_processor.py`
- `src/llm_values/llm_pca_analysis.py`
- `src/base/base_interview.py`
- `src/base/ivs_question_processor.py`

In this pipeline:

1. `llm_interview.py` sends the ten IVS/WVS items to each model.
2. `base_interview.py` loads API settings from `.env`, instantiates the model
   clients declared in `config/models/llm_models.json`, enforces numeric answer
   formatting, and retries malformed responses.
3. `llm_data_processor.py` converts the collected answers into IVS-compatible
   fields.
4. `ivs_question_processor.py` applies the shared recoding logic for `Y002` and
   `Y003`.
5. `llm_pca_analysis.py` projects processed model responses into the fixed
   benchmark space.

The original private workflow wrote raw response caches under
`data/llm_values/` or `data/llm_interviews/intrinsic/interview_raw/`. Those
full raw caches are not required for reproducing the published results and are
not part of the public Dryad package.

### Multilingual intrinsic pipeline

The multilingual intrinsic pipeline queries each model in the six UN languages
and generates the language-comparison coordinates used in the baseline language
analysis.

Run:

```bash
python src/run/run_llm_multilingual_analysis.py --step all
```

Core files:

- `src/run/run_llm_multilingual_analysis.py`
- `src/llm_values/llm_multilingual_interview.py`
- `src/llm_values/llm_multilingual_data_processor.py`
- `src/base/base_interview.py`
- `src/base/ivs_question_processor.py`

Useful commands:

```bash
python src/run/run_llm_multilingual_analysis.py --step 1 --consensus-count 5
python src/run/run_llm_multilingual_analysis.py --step 2
python src/run/run_llm_multilingual_analysis.py --step 3
```

Inputs:

- `config/questions/multilingual/multilingual_questions_complete.json`
- API keys in `.env` if private reruns are desired
- released processed tables in `data/llm_interviews/intrinsic/`

Outputs:

- `data/llm_interviews/intrinsic/multilingual_ivs_format.pkl`
- `data/llm_interviews/intrinsic/multilingual_ivs_format.json`
- `data/llm_pca/intrinsic/llm_pca_entity_scores.pkl`
- `data/llm_pca/intrinsic/llm_pca_entity_scores.json`

### Multilingual country-roleplay pipeline

This pipeline produces the country-language roleplay coordinates used in the
English-advantage, regional, and East Asia analyses.

Run:

```bash
python src/run/run_roleplay_multilingual_analysis.py
```

Core files:

- `src/run/run_roleplay_multilingual_analysis.py`
- `src/roleplay_multilingual/multilingual_roleplay_interview.py`
- `src/roleplay_multilingual/multilingual_roleplay_data_processor.py`
- `src/roleplay_multilingual/multilingual_roleplay_pca_analysis.py`
- `src/base/base_interview.py`
- `src/base/ivs_question_processor.py`

In this pipeline:

1. `multilingual_roleplay_interview.py` constructs country-language roleplay
   prompts for each model.
2. `base_interview.py` enforces numeric outputs and retry logic.
3. `multilingual_roleplay_data_processor.py` validates and recodes responses and
   standardizes country-language metadata.
4. `multilingual_roleplay_pca_analysis.py` projects the processed answers into
   the fixed benchmark space.

As with the intrinsic workflow, the full private `interview_raw/` caches are
not needed for public reproduction and are not included in the Dryad package.
The released archive instead provides the processed interview tables and the
projected coordinate files used in the paper.

## Shared Validation And Recoding

Two modules define the common response-processing logic used across the
repository.

### `src/base/base_interview.py`

This module handles:

- API key loading from `.env`
- model-client creation from `config/models/llm_models.json`
- language-specific numeric format hints
- retry logic for malformed or incomplete answers
- raw-answer collection during private data collection

During private reruns, failed or malformed model responses are retried
automatically. If retries are exhausted, the error is reported in the console
output and the corresponding pipeline step stops with an exception.

### `src/base/ivs_question_processor.py`

This module handles:

- extracting numeric values from answer text
- validating the expected format for each IVS item
- recoding `Y002` into the materialist/postmaterialist index
- recoding `Y003` into the binary child-quality indicators and the derived
  traditional-versus-secular-rational score

These rules are shared across the intrinsic and roleplay pipelines so that all
released tables enter the downstream analyses in a common format.

## Figure Reproduction

This repository includes direct rendering scripts for Main Text Figure 2 and
Supplementary Figures S2-S5. Main Text Figures 1, 3, and 4, together with
Supplementary Figure S1, are backed by released source tables but are not
currently regenerated by standalone scripts in this cleaned public repository.
For those figures, the final manuscript layouts were assembled outside the
cleaned public code release; no additional statistical computation beyond the
released source tables is introduced in those final assemblies.

| Figure | Script | Main released source data | Output location |
| --- | --- | --- | --- |
| Main Text Figure 1 | Released source data only; no standalone public renderer | `Supplementary Materials/data/figure1_data_66countries.csv` | Final manuscript assembly |
| Main Text Figure 2 | `python src/figures/generate_fig2_baseline.py` | `Supplementary Materials/data/figure2_baseline_20models.csv`, `Supplementary Materials/data/DataS1_ivs_pca_coordinates.csv` | `results/figures/paper/` |
| Main Text Figure 3 | Released source data only; no standalone public renderer | `Supplementary Materials/data/figure3_digital_orientalism.csv`, `Supplementary Materials/data/study3_digital_orientalism.json` | Final manuscript assembly |
| Main Text Figure 4 | Released source data only; no standalone public renderer | `Supplementary Materials/data/figure4_colonial_history.csv`, `Supplementary Materials/data/study4_colonial_legacies.json` | Final manuscript assembly |
| Supplementary Figure S1 | Released source data only; final figure assembled manually during manuscript preparation | `Supplementary Materials/data/DataS2_llm_baseline_pca.csv` | Final manuscript assembly |
| Supplementary Figure S2 | `python src/figures/generate_figS2_model_imitation.py` | `Supplementary Materials/data/model_imitation_accuracy.csv`, `Supplementary Materials/data/study5_model_imitation.json` | `Supplementary Materials/figures/FigS2/` |
| Supplementary Figure S3 | `python src/figures/generate_figS3_ivs_cultural_map.py` | `Supplementary Materials/data/DataS1_ivs_pca_coordinates.csv` or `Supplementary Materials/data/ivs_pca_coordinates.csv` | `Supplementary Materials/figures/FigS3/` |
| Supplementary Figure S4 | `python src/figures/generate_figS4_english_advantage.py` | `Supplementary Materials/data/figure3_digital_orientalism.csv`, `Supplementary Materials/data/study3_digital_orientalism.json` | `Supplementary Materials/figures/FigS4/` |
| Supplementary Figure S5 | `python src/figures/generate_figS5_east_asia.py` | `Supplementary Materials/data/llm_roleplay_pca.csv`, `Supplementary Materials/data/ivs_pca_coordinates.csv` | `Supplementary Materials/figures/FigS5/` |

All figure scripts read publication-ready source tables from
`Supplementary Materials/data/`.

## Environment Setup

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

If you want to rerun private API calls rather than use the released processed
tables, create a `.env` file in the repository root and provide the keys
referenced in `config/models/llm_models.json`.

## License

The code in this repository is released under the MIT License. This license
applies to the software and documentation in the GitHub repository only. The
processed data released through Dryad should be interpreted according to the
terms of the Dryad record and the associated manuscript.

## Citation

If you use this repository or the associated released data, please cite the
paper and the archival data/code records associated with the public release.
A compact citation entry can be added here once the final manuscript citation
and repository DOIs are available.

## Notes

- The public release is designed for reproducibility of the reported analyses,
  not for redistribution of raw commercial API-response caches.
- The raw IVS/WVS/EVS file is not included in this repository.
- In normal use, this repository should be paired with the Dryad data package
  rather than treated as a standalone data archive.
