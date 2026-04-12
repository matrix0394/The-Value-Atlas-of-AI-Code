Place the publication-ready supplementary source data from Dryad in this
directory before running the figure-generation scripts in `src/figures/`.

Required files by script:

- `src/figures/generate_fig2_baseline.py`
  - `figure2_baseline_20models.csv`
  - `DataS1_ivs_pca_coordinates.csv` or `ivs_pca_coordinates.csv`
- `src/figures/generate_figS2_model_imitation.py`
  - `DataS1_ivs_pca_coordinates.csv` or `ivs_pca_coordinates.csv`
  - `DataS2_llm_baseline_pca.csv` or `llm_baseline_pca.csv`
  - `DataS3_llm_roleplay_pca.csv` or `llm_roleplay_pca.csv`
  - `model_imitation_accuracy.csv`
  - `study5_model_imitation.json`
- `src/figures/generate_figS3_ivs_cultural_map.py`
  - `DataS1_ivs_pca_coordinates.csv` or `ivs_pca_coordinates.csv`
- `src/figures/generate_figS4_english_advantage.py`
  - `figure3_digital_orientalism.csv`
  - `study3_digital_orientalism.json`
- `src/figures/generate_figS5_east_asia.py`
  - `DataS1_ivs_pca_coordinates.csv` or `ivs_pca_coordinates.csv`
  - `DataS3_llm_roleplay_pca.csv` or `llm_roleplay_pca.csv`

Additional publication-ready tables released with the paper and included in the
Dryad package:

- `DataS2_llm_baseline_pca.csv`
- `DataS3_prompts.json`
- `figure1_data_66countries.csv`
- `figure2_language_summary.csv`
- `figure4_colonial_history.csv`
- `regression_data.csv`
- `paper_statistics_all.json`
- `study1_intrinsic_bias.json`
- `study2_english_advantage.json`
- `study4_colonial_legacies.json`
