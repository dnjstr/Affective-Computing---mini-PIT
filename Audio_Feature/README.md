# Audio feature pipeline

Install the root `requirements.txt`, then run `python Audio_Feature/audio_feature_pipeline.py` from the repository root (or use the script's absolute path). Inputs resolve relative to the script, not the current working directory.

`output/features_raw.csv` preserves reference metadata and QC. `features_unimputed.csv` preserves missing predictor values for training-fold preprocessing. The two `features_eda_*.csv` exports are for visualization only. Errors are recorded in `extraction_errors.csv`; the script stops instead of silently imputing failed recordings.

See [methodology](../docs/METHODOLOGY.md) and the root README for fusion and evaluation.
