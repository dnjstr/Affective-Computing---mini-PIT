import pathlib
import sys
import importlib.util

project_root = pathlib.Path(__file__).resolve().parents[1]
audio_feature_dir = project_root / "Audio_Feature"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(audio_feature_dir))

import soundfile as sf
import librosa

module_path = audio_feature_dir / "audio_feature_pipeline.py"
spec = importlib.util.spec_from_file_location("audio_feature_pipeline", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load audio_feature_pipeline from {module_path}")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

path = audio_feature_dir / "audio" / "Y4D-009_Y4_PM_Grabe.wav"
raw, sr = sf.read(str(path), always_2d=True);            print("raw", raw.shape, sr)
y, _ = librosa.load(str(path), sr=a.SR, mono=True);      print("16k mono", y.shape)
yt, _ = librosa.effects.trim(y, top_db=a.TRIM_DB);      print("trimmed", yt.shape)
f = a.extract_features(str(path));                       print("vector", len(f), "values")