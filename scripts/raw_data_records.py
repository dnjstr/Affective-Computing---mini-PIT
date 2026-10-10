import pandas as pd
from pathlib import Path
m = pd.read_csv("Audio_Feature/G4_-_RESEARCH_MINI-PROJECT.csv", encoding="utf-8-sig")
wav = {p.name for p in Path("Audio_Feature/audio").glob("*.wav")}
mp4 = {p.name for p in Path("Facial_Feature/video").glob("*.mp4")}
print("rows:", len(m), "| unique IDs:", m["Participant Code"].is_unique)
print("wav:", len(wav), "| mp4:", len(mp4))
print("missing wav:", set(m["Audio Filename"]) - wav, "| missing mp4:", set(m["Video Filename"]) - mp4)
print("consent:", m["Consent Obtained"].value_counts().to_dict())