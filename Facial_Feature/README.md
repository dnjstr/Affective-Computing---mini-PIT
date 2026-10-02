# Facial feature pipeline

Install the root `requirements.txt`, then run `python Facial_Feature/face_feature_pipeline.py`. Videos, metadata and `face_landmarker.task` are resolved relative to this script.

The raw export retains all metadata rows; failed/low-quality samples are tracked in diagnostics. `face_features_unimputed.csv` includes only QC-passing recordings and excludes duplicated composites and the neutral coefficient. EDA imputation/scaling is explicitly named `face_features_eda_*.csv`.

Blendshape and action-unit-style composites are expression proxies. Read [methodology](../docs/METHODOLOGY.md) before interpreting them as emotion features.
