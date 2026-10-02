# Methodology

## Audio

Original-channel clipping is checked before resampling and mono conversion. Every recording is loaded at 16 kHz; leading/trailing low-energy material is trimmed using a 30 dB threshold. A 2,048-sample frame and 512-sample hop are used consistently for the principal spectral features, MFCCs and pYIN pitch (65–500 Hz). These are 128 ms frames and 32 ms hops: useful for broad summaries, but relatively coarse for short utterances and transient dynamics.

Pitch summaries use finite voiced estimates. The slope fits voiced F0 against its original frame timestamps, preserving intervening unvoiced gaps. The period-variation proxy includes only adjacent voiced frames. Neither this proxy nor RMS frame variation is a clinical cycle-level jitter/shimmer measurement. Onset rate is an acoustic activity proxy, not a measured syllable rate. Six missing-pitch recordings in the original collection should not be interpreted automatically as silent recordings.

Other descriptors include RMS, zero crossing rate, centroid, bandwidth, rolloff, flatness, pooled spectral contrast/chroma summaries, and 13 MFCCs with first/second deltas. Means and standard deviations summarize each MFCC independently; contrast/chroma are pooled descriptors, not full band/bin vectors. Delta width adapts for short clips. Very short signals are padded; duration metadata records the unpadded interval.

Pitch range and single-word clips limit F0 reliability. A value near the search ceiling should be inspected, not automatically labeled correct or discarded.

## Facial

OpenCV decodes frames and converts BGR to RGB. A fresh MediaPipe VIDEO-mode Face Landmarker processes each recording with increasing millisecond timestamps, one face and blendshape/transformation outputs. All frames are processed. Facial geometry uses 2D pixel coordinates; eye and mouth aspect ratios normalize distances by eye/mouth width but remain affected by pose and perspective.

Blendshapes are summarized by mean, population standard deviation and maximum. Additional composites include bilateral smile/frown/brow lowering, eye squint, and a disgust-expression proxy. Head rotation is derived from the facial transformation matrix after column scale normalization; angle signs are coordinate conventions, not emotion directions. Validate orientation against known head poses before interpreting signs clinically.

Temporal variability and frame-to-frame absolute blendshape change are expressiveness/motion proxies. Motion energy is per frame, not per second; all current saved clips report 30 fps. With other frame rates or detection gaps, adapt/resample before comparing dynamic features. A minimum face detection rate of 80% defines eligibility. Failed-quality rows stay in the raw/QC tables and are excluded from predictor exports. Detection does not verify identity or landmark accuracy.

## Fusion and quality

Join by unique participant_code using one-to-one validation. Shared metadata must agree; conflicting labels and duplicate IDs raise an error. An outer raw join preserves unmatched rows. The predictor export requires both modalities, successful audio extraction, and passing facial QC. Pitch may remain missing in an otherwise usable sample.

Do not infer synchronization from matching IDs. Audio/video durations are recorded and absolute discrepancies greater than 0.1 seconds are flagged for review. This threshold is a review heuristic, not an exclusion rule or proof of recording corruption.

## Statistical and model analysis

Descriptive statistics and Spearman tests operate on observed values. Multimodal feature–target, representative cross-modal, and performed Kruskal tests receive Benjamini–Hochberg correction in separate test families. Group tests are skipped when any group has fewer than two observed values; therefore the current singleton Negative-valence class prevents a valid three-group valence comparison in this implementation. Original modality p-values are also corrected separately for each modality/test family.

PCA plots globally impute medians, discard mostly missing/constant features and standardize; their sole purpose is EDA. The optional baseline uses shuffled five-fold KFold (seed 41), training-fold median imputation with missing indicators, constant-feature removal, scaling, selection of 10 predictors by regression F score, and Ridge(alpha=10). The dummy model predicts the training mean. MAE/RMSE describe out-of-fold numeric-score error. Targets are ordinal, so treating them as numeric is an exploratory approximation. Hyperparameters are fixed, not tuned on these results. Do not select the best approach from this dataset and call its same-data score an unbiased final estimate.

Sources: [librosa pYIN](https://librosa.org/doc/0.11.0/generated/librosa.pyin.html), [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python), [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
