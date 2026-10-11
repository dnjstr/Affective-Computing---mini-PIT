"""Exploratory context analysis and lexically controlled comparison (protocol sections XIII and XVII).

Reads:
  Audio_Feature/G4_-_RESEARCH_MINI-PROJECT.csv                      (metadata: year, activity, session, word)
  Multimodal_Feature/output_multimodal/multimodal_features_unimputed.csv   (labels and acoustic features)
Writes to Multimodal_Feature/output_multimodal/exploratory/:
  context_affect_counts.csv   counts of each context value by valence group and arousal group
  context_affect_means.csv    n, mean valence score and mean arousal score for each context value
  lexical_control.csv         every recording of a word that was spoken at least twice, with acoustic features

"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "Audio_Feature" / "G4_-_RESEARCH_MINI-PROJECT.csv"
FEATS = ROOT / "Multimodal_Feature" / "output_multimodal" / "multimodal_features_unimputed.csv"
OUT = ROOT / "Multimodal_Feature" / "output_multimodal" / "exploratory"
OUT.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(META, encoding="utf-8-sig")
meta.columns = [c.strip() for c in meta.columns]
ctx = pd.DataFrame({
    "participant_code": meta["Participant Code"].str.strip(),
    "Year Level": meta["Year Level"].str.strip(),
    "Class Activity": meta["Class Activity"].str.strip(),
    "Session": meta["Session"].str.strip(),
    "Spoken Word": meta["Spoken Word"].str.strip().str.title(),
})
feats = pd.read_csv(FEATS)
df = ctx.merge(feats, on="participant_code", how="inner", validate="one_to_one")
assert len(df) == len(ctx), "Some participants are missing from the feature table"

CONTEXTS = ["Year Level", "Class Activity", "Session", "Spoken Word"]
TARGET_GROUPS = ["valence_group", "arousal_group"]

# Section XIII: context x affect
count_rows, mean_rows = [], []
for c in CONTEXTS:
    for g in TARGET_GROUPS:
        tab = pd.crosstab(df[c], df[g])
        for value, row in tab.iterrows():
            for group, n in row.items():
                count_rows.append({"context": c, "value": value, "target": g,
                                   "group": group, "n": int(n)})
    m = df.groupby(c).agg(n=("participant_code", "count"),
                          mean_valence=("valence_score", "mean"),
                          mean_arousal=("arousal_score", "mean")).round(2)
    for value, row in m.iterrows():
        mean_rows.append({"context": c, "value": value, "n": int(row["n"]),
                          "mean_valence": row["mean_valence"], "mean_arousal": row["mean_arousal"]})
pd.DataFrame(count_rows).to_csv(OUT / "context_affect_counts.csv", index=False)
means = pd.DataFrame(mean_rows)
means.to_csv(OUT / "context_affect_means.csv", index=False)

print("=== Context x affect (n, mean valence, mean arousal) ===")
for c in CONTEXTS:
    if c == "Spoken Word":
        continue
    print(f"\n{c}")
    print(means[means.context == c].drop(columns="context").to_string(index=False))
    for g in TARGET_GROUPS:
        print(pd.crosstab(df[c], df[g]).to_string())
print("\nSpoken Word: see context_affect_counts.csv and context_affect_means.csv")

# Section XVII: lexically controlled comparison
ACOUSTIC = ["audio_f0_mean_hz", "audio_f0_std_semitones", "audio_rms_mean",
            "audio_zcr_mean", "audio_onset_rate_per_sec"]
ACOUSTIC = [c for c in ACOUSTIC if c in df.columns]
word_n = df["Spoken Word"].value_counts()
repeated = word_n[word_n >= 2].index
lex = df[df["Spoken Word"].isin(repeated)].copy()
lex["valence_differs_within_word"] = lex.groupby("Spoken Word")["valence_score"].transform("nunique") > 1
lex["arousal_differs_within_word"] = lex.groupby("Spoken Word")["arousal_score"].transform("nunique") > 1
cols = ["Spoken Word", "participant_code", "valence_score", "arousal_score",
        "valence_group", "arousal_group", "valence_differs_within_word",
        "arousal_differs_within_word"] + ACOUSTIC
lex = lex[cols].sort_values(["Spoken Word", "participant_code"])
lex.to_csv(OUT / "lexical_control.csv", index=False)

print("\n=== Words spoken at least twice ===")
print(word_n[word_n >= 2].to_string())
print("\n=== Lexically controlled comparison (same word, different valence) ===")
diff = lex[lex["valence_differs_within_word"]]
if diff.empty:
    print("No repeated word has different valence scores.")
else:
    print(diff.round(3).to_string(index=False))
print("\nNote: counts per word are very small, so this comparison is descriptive only.")
print(f"\nSaved files in {OUT}")