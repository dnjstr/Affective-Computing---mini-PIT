# Multimodal early fusion

After both extraction stages, run:

```bash
python Multimodal_Feature/multimodal_pipeline.py
python scripts/evaluate_baselines.py
```

The first command joins raw tables by participant ID, checks metadata consistency, exports missing-preserving fusion predictors and EDA copies, and generates six plots plus FDR-adjusted exploratory statistics. The second generates a seventh plot comparing audio, facial and combined regression against a training-mean baseline.

Use `output_multimodal/multimodal_features_unimputed.csv` with `feature_manifest.csv` for evaluation. See `output_multimodal/RESULTS.md` for sample counts and interpretation. No temporal synchronization is assumed.
