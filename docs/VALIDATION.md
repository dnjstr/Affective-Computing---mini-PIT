# Validation and change record

Base repository: `dnjstr/Affective-Computing---mini-PIT`, commit `795da9ab7d913d960e607c63f296e87371c31c27`. Prepared 2026-10-02.

## Implemented changes

- Resolve audio input/output paths relative to the script.
- Reject invalid labels and missing recording paths.
- Correct F0 slope timestamps and prevent period-variation calculations across unvoiced gaps.
- Measure clipping on original audio channels, before resampling/mono averaging.
- Preserve extraction errors; do not fabricate successful audio samples by median imputation.
- Exclude failed facial QC from predictor exports.
- Remove duplicate facial composite aliases from predictor selection.
- Export raw unimputed predictors and explicitly named EDA-only transformed copies.
- Preserve failed/unmatched multimodal rows in raw/QC exports while requiring complete, QC-passing pairs for fusion predictors.
- Validate unique join IDs and consistency of shared metadata/targets.
- Add descriptive summaries, missingness, FDR-adjusted statistics, six combined EDA plots and one exploratory out-of-fold model comparison.
- Avoid silently dropping singleton groups in Kruskal comparisons.
- Document feature definitions, safe evaluation and limitations.

## Executed checks

Audio extraction was rerun on all 25 original WAVs using the revised code and pinned root requirements. Every recording completed. Facial extraction was then rerun on all 25 original MP4s with user approval on 2026-10-02; every recording completed, with face detection on 100% of decoded frames and all samples passing the 80% quality criterion. Both modality predictor exports/statistical tables/plots and the multimodal outputs are regenerated. No change has been pushed to GitHub.

Initial facial reruns were stopped by automatic approval review because the installed MediaPipe package attempted contact with `play.googleapis.com/log`. Before the successful retry, the installed binary and official MediaPipe privacy notice were examined. The notice states that video/image inputs are processed on-device and are not sent to Google servers, while performance and utilization metrics are sent to Google. This documented behavior, together with user approval, allowed the retry. This is not an independent packet-level verification of the request payload. The root README includes the metrics notice: https://github.com/google-ai-edge/mediapipe#privacy-notice.

Seven targeted tests passed: ID-based pairing despite shuffled rows, rejection of conflicting metadata/duplicate IDs, predictor exclusion of targets/QC/aliases, preservation of raw missing values, BH correction, corrected pitch slope and adjacent-voicing calculation, and failed-facial-QC exclusion. Python compilation passed. The multimodal pipeline and exploratory evaluation completed on 25 paired samples.

The evaluation uses 1–5 numeric targets rather than claiming reliable three-class performance with a singleton Negative-valence class. Training transformations and supervised feature selection are fitted within folds. Report results as exploratory only: the same small dataset informs all comparisons, features are high-dimensional, and there is no independent test set.

## Remaining measurement checks

- Listen to and inspect pitch tracks, especially `Y4D-020` near the configured upper bound.
- Visually inspect landmark/expression tracking on source videos.
- Verify that WAV/MP4 intervals correspond before making synchronization claims.
- Validate head-pose sign conventions against known poses.
- Account for the documented MediaPipe API metrics behavior when running on recordings.
