# Facial Feature Extraction

Put this folder next to `Audio_Feature`, then inside it add:
- `Videos/`  (the 25 .mp4 files)
- `G4_-_RESEARCH_MINI-PROJECT.csv`

```bash
python -m pip install -r requirements.txt
python Extract_Facial_Feature.py   # -> facial_feature_dataset.csv, facial_frame_features.csv
```

`mediapipe` is pinned to 0.10.14 because newer versions removed the `mp.solutions.face_mesh` API used here.