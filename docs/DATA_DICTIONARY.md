# Data dictionary

`feature_manifest.csv` is the authoritative list of exported candidate predictors, including modality, missing count and whether each feature is constant in this particular collection. Constants are retained in the unimputed export; select/remove them using training data.

| Field / family | Meaning | Evaluation use |
|---|---|---|
| participant_code | Unique join key | Identifier only |
| valence_score, arousal_score | Self-reported integer 1–5 targets | Choose the required target |
| valence_group, arousal_group, affect_category | Derived target categories | Targets only; never predictors |
| year_level, class_activity, subject, session_name, date_collected | Collection context | Metadata; excluded from predictors |
| spoken_word, self_reported_feeling, filenames, consent | Lexical/reference/collection data | Metadata; excluded from predictors |
| audio_f0_* | Pitch summaries, Hz/semitones/Hz per second | Numeric predictors; may be missing |
| audio_period_variation_proxy | Mean relative F0 change between adjacent voiced frames | Proxy predictor; not clinical jitter |
| audio_rms_* | Frame RMS summaries and dB/variation descriptors | Numeric predictors; not calibrated sound pressure |
| audio_voiced_ratio, audio_active_ratio | Voiced-frame and energy-active ratios | Numeric predictors; both also inform QC |
| audio_onset_rate_per_sec | Detected onsets divided by trimmed/padded duration | Acoustic activity proxy |
| audio_zcr_*, audio_spec_*, audio_chroma_* | Spectral/acoustic summaries | Numeric predictors |
| audio_mfcc_*, audio_delta_mfcc_*, audio_delta2_mfcc_* | 13 coefficient-wise MFCC/delta mean/std pairs | Numeric predictors |
| audio_trimmed_dur_sec | Clip duration after trim, before padding | Numeric timing predictor |
| audio_extraction_ok, audio_pitch_detected, audio_clipping_ratio, sample rates, channel count, frame counts, delta availability | Extraction/QC diagnostics | Excluded from model predictors |
| face_<blendshape>_<mean/std/max> | Temporal summaries of model expression coefficients | Numeric predictors; expression proxies |
| face_smile_*, face_frown_*, face_brow_lowerer_* | Bilateral composite summaries | Numeric proxy predictors |
| face_ear_*, face_mar_* | Eye/mouth aspect-ratio summaries | Numeric geometric predictors |
| face_pitch/yaw/roll_*_deg | Rotation means/std in degrees | Numeric predictors; coordinate-dependent |
| face_expressiveness_proxy, face_motion_energy | Temporal variability/change | Numeric proxy predictors |
| face_detection_rate, face_detected_frames, face_detection_quality_pass, video_* | Quality and recording diagnostics | Excluded from model predictors |

The duplicate `face_brow_inner_raiser_*` and `face_jaw_open_*` aliases remain documented in raw measurements but are excluded from predictor exports. Use their original `face_browInnerUp_*` and `face_jawOpen_*` counterparts. `_neutral` blendshape coefficients are excluded from predictors.

`multimodal_features_unimputed.csv` combines candidate predictors plus seven identifier/target/context columns. Do **not** use every numeric column as X: valence/arousal scores are included for convenience. Use the exact `feature` values in the manifest.

Missing values remain NaN/blank in unimputed exports. EDA-only imputed/scaled exports have globally fitted transformations. The optional evaluation script reads the unimputed export and performs all learned preprocessing inside cross-validation.
