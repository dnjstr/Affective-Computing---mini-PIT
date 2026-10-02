"""Run all extraction and fusion stages from any working directory."""
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for path in ['Audio_Feature/audio_feature_pipeline.py','Facial_Feature/face_feature_pipeline.py',
             'Multimodal_Feature/multimodal_pipeline.py']:
    subprocess.run([sys.executable,str(root/path)],cwd=root,check=True)
