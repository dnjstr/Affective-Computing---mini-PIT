# Multimodal results

Generated from audio and facial raw feature tables. See `docs/VALIDATION.md` for the delivered run provenance.

- Participants available: 25; complete QC-passing pairs: 25.
- Unfiltered numeric predictors: 113 audio + 178 facial = 291.
- Pitch missing in 6 recordings. Missing pitch is retained in the unimputed export.
- Duration review flags (>0.1 s): 21; these flags do not establish misalignment.
- Feature–target tests: 582; q < 0.05 after BH correction: 0.

## Interpretation

This is participant-level early fusion: numeric audio and facial features are concatenated by participant_code, with one target set. It is not frame-level synchronization. Each modality covers its own clip interval.

Use `multimodal_features_unimputed.csv` and `feature_manifest.csv` for evaluation. Fit imputation, constant-feature removal, scaling and any feature selection only on the training fold. The EDA exports and PCA plots use full-dataset transformations and are not evaluation inputs.

Self-reported valence/arousal are targets. Spoken words, filenames, year level, IDs, quality indicators, and other target representations must not become predictors. Blendshapes and composites are expression proxies, not diagnostic emotion labels; pitch/energy variation are not clinical jitter/shimmer.

There are only 25 original recordings, with imbalanced targets and more predictors than samples. Correlations are exploratory, not causal. Valence group tests are skipped if any class has fewer than two observed samples, rather than silently dropping that class. BH corrections are applied separately across all feature–target tests, all performed group tests, and the fixed representative cross-modal pairs. Dependence among features and small samples still limit inference. Year, class activity, subject and collection date are confounded in this collection.

## Figures

1. Target distributions and valence–arousal table.
2. Per-participant audio/facial quality and duration comparison.
3. Predictor missingness before imputation.
4. Cross-modal Spearman heatmap for a fixed representative feature set.
5. Strongest observed feature–target correlations (not a significance claim).
6. Audio, facial and combined PCA projections (EDA only).
