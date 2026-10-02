# Affective Computing mini-PIT: audio, facial and multimodal features

Extract participant-level acoustic/prosodic and facial-expression features from paired short recordings, join them safely, and generate exploratory statistics and plots.

## Quick start

Use Python 3.12 (the tested version), preferably in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_all.py
python scripts/evaluate_baselines.py
python scripts/measurement_checks.py
python -m unittest discover -s tests -v
```

Run scripts from any working directory. Keep the metadata CSVs, WAVs, MP4s and `face_landmarker.task` in their existing modality folders. On Linux, MediaPipe may additionally require system OpenGL/GLES libraries; install `libgles2` and `libegl1` using your distribution's package manager if those shared libraries are missing. The model is included; downloading it is only attempted if it is absent.

## MediaPipe metrics

MediaPipe’s [official privacy notice](https://github.com/google-ai-edge/mediapipe#privacy-notice) states that input video is processed on-device and is not sent to Google servers, while API performance and utilization metrics are sent to Google. The tested package contains an HTTPS logging endpoint at `play.googleapis.com/log`. Account for this behavior before running it on recordings; this project does not upload video through its own code. This notice describes documented behavior, not a packet-level audit.

## Outputs

| Location | Contents |
|---|---|
| `Audio_Feature/output/` | Raw audio measurements, unimputed predictors, EDA exports and original modality plots |
| `Facial_Feature/output_face/` | Raw facial measurements, QC-passing unimputed predictors, EDA exports and modality plots |
| `Multimodal_Feature/output_multimodal/` | Joined raw data, fusion predictors, sample QC, manifest, FDR statistics, seven combined figures and results Markdown |

For evaluation, use `*_unimputed.csv`, explicitly select predictors from the manifest, and fit preprocessing inside each training fold. `*_eda_imputed.csv` and `*_eda_scaled.csv` use the whole dataset for visualization. Legacy `*_clean_unscaled.csv` and `*_model_ready_scaled.csv` are removed during regeneration to avoid ambiguous use.

The combined dataset is **participant-level early fusion**, not synchronized frame-level fusion. Self-reported valence/arousal are the targets; expression coefficients are not ground-truth emotion labels.

## Documentation

- [Methodology and feature definitions](docs/METHODOLOGY.md)
- [Data dictionary and safe predictor selection](docs/DATA_DICTIONARY.md)
- [Validation and changes](docs/VALIDATION.md)
- [Measurement checks report](docs/MEASUREMENT_CHECKS.md)
- [Generated multimodal results](Multimodal_Feature/output_multimodal/RESULTS.md)

This collection has only 25 recordings, substantial target imbalance, and more features than samples. All statistical relationships and model comparisons should be reported as exploratory. The optional comparison predicts numeric 1–5 scores using the same five folds for each modality and a dummy baseline; it does not provide an independent test-set result.
