import re
import sys
import zipfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# ----------------------------------------------------------------------------
# CONFIG  (edit these paths)
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
CSV_PATH = BASE / "G4_-_RESEARCH_MINI-PROJECT.csv"
AUDIO_DIR = BASE / "audio"
ZIP_PATH = None  # optionally set to an audio ZIP
OUT_DIR = BASE / "output"
FIG_DIR = OUT_DIR / "figures"

SR = 16000            # sampling rate used for ALL files
N_FFT = 2048          # frame length (same as reference group)
HOP = 512
TRIM_DB = 30          # silence trimming threshold (dB below peak)
F0_MIN, F0_MAX = 65.0, 500.0   # pitch search range (Hz) for pyin
N_MFCC = 13

OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# STEP 1 — METADATA CLEANING
# ----------------------------------------------------------------------------
def unzip_audio():
    if ZIP_PATH and Path(ZIP_PATH).exists():
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(AUDIO_DIR)
        print(f"[1] Extracted {ZIP_PATH} -> {AUDIO_DIR}/")


def _num(s):
    """'5 — Very Pleasant' -> 5"""
    m = re.match(r"\s*(\d)", str(s))
    return int(m.group(1)) if m else np.nan


VAL_LABEL = {1: "Very Unpleasant", 2: "Unpleasant", 3: "Neutral", 4: "Pleasant", 5: "Very Pleasant"}
ARO_LABEL = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}


def group_valence(v):
    return "Negative" if v <= 2 else ("Neutral" if v == 3 else "Positive")


def group_arousal(a):
    return "Low" if a <= 2 else ("Moderate" if a == 3 else "High")


def load_metadata():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all")

    out = pd.DataFrame({
        "participant_code": df["Participant Code"].str.strip(),
        "year_level": df["Year Level"].str.strip(),
        "class_activity": df["Class Activity"].str.strip(),
        "subject": df["Subject"].str.strip(),
        "session_name": df["Session"].str.strip(),
        "spoken_word": df["Spoken Word"].str.strip(),     # kept for reference only, NOT a feature
        "self_reported_feeling": df["Self-Reported Feeling"].str.strip(),
        "valence_score": df["Valence (1-5)"].map(_num),
        "arousal_score": df["Arousal (1-5)"].map(_num),
        "date_collected": df["Date Collected"],
        "consent_obtained": df["Consent Obtained"].str.strip().str.lower().eq("yes"),
        "audio_filename": df["Audio Filename"].str.strip(),
    })
    out["valence_label"] = out["valence_score"].map(VAL_LABEL)
    out["arousal_label"] = out["arousal_score"].map(ARO_LABEL)
    out["valence_group"] = out["valence_score"].apply(group_valence)
    out["arousal_group"] = out["arousal_score"].apply(group_arousal)
    out["affect_category"] = out["valence_group"] + "-" + out["arousal_group"]

    # --- QC checks -----------------------------------------------------------
    assert out["participant_code"].is_unique, "Duplicate participant codes!"
    n_noconsent = (~out["consent_obtained"]).sum()
    if n_noconsent:
        print(f"[1] WARNING: {n_noconsent} rows without consent -> excluded")
        out = out[out["consent_obtained"]]

    if not out["valence_score"].isin(range(1, 6)).all() or not out["arousal_score"].isin(range(1, 6)).all():
        raise ValueError("Valence/arousal labels must be integers from 1 to 5")

    # Match audio files case-insensitively (.WAV vs .wav)
    lookup = {p.name.lower(): p for p in AUDIO_DIR.rglob("*") if p.suffix.lower() == ".wav"}
    out["audio_path"] = out["audio_filename"].str.lower().map(lookup)
    missing = out[out["audio_path"].isna()]
    if len(missing):
        print("[1] WARNING: audio file not found for:", missing["participant_code"].tolist())
        raise FileNotFoundError("Missing audio files: " + ", ".join(missing["participant_code"]))

    # does the filename word match the spoken-word column?
    out["metadata_word_matches_filename"] = [
        str(w).lower() == str(f).split("_")[-1].rsplit(".", 1)[0].lower()
        for w, f in zip(out["spoken_word"], out["audio_filename"])
    ]
    print(f"[1] Metadata ready: {len(out)} usable recordings")
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# STEP 2/3 — PREPROCESSING + FEATURE EXTRACTION
# ----------------------------------------------------------------------------
def summ(prefix, x, feats, mean=True, std=True):
    """add mean/std of a 1-D array to dict"""
    if mean:
        feats[f"{prefix}_mean"] = float(np.mean(x))
    if std:
        feats[f"{prefix}_std"] = float(np.std(x))


def extract_features(path):
    f = {}

    # ---- load & preprocess ---------------------------------------------------
    import soundfile as sf
    original, original_sr = sf.read(path, always_2d=True)
    if not original.size or not np.isfinite(original).all():
        raise ValueError("Empty or non-finite audio")
    f["audio_original_sampling_rate"] = original_sr
    f["audio_original_channels"] = original.shape[1]
    f["audio_clipping_ratio"] = float(np.mean(np.abs(original) >= 0.99))
    y_raw, _ = librosa.load(path, sr=SR, mono=True)
    f["audio_duration_sec"] = len(y_raw) / SR
    y, idx = librosa.effects.trim(y_raw, top_db=TRIM_DB)   # remove leading/trailing silence
    f["audio_trimmed_dur_sec"] = len(y) / SR
    f["audio_trimmed_fraction"] = len(y) / max(len(y_raw), 1)
    f["audio_sampling_rate"] = SR
    if np.max(np.abs(y_raw)) < 1e-8:
        raise ValueError("Silent recording")

    if len(y) < N_FFT:                       # pad very short clips so STFT works
        y = np.pad(y, (0, N_FFT - len(y)))

    # ---- PROSODIC: pitch (F0) -----------------------------------------------
    f0, voiced_flag, _ = librosa.pyin(y, fmin=F0_MIN, fmax=F0_MAX, sr=SR,
                                      frame_length=N_FFT, hop_length=HOP)
    valid = np.isfinite(f0)
    f0v = f0[valid]
    f["audio_pitch_detected"] = bool(len(f0v) > 0)
    f["audio_voiced_frames"] = int(len(f0v))
    f["audio_pitch_total_frames"] = int(len(f0))
    f["audio_voiced_ratio"] = len(f0v) / max(len(f0), 1)
    if len(f0v) >= 3:
        f["audio_f0_mean_hz"] = float(np.mean(f0v))
        f["audio_f0_std_hz"] = float(np.std(f0v))
        f["audio_f0_min_hz"] = float(np.min(f0v))
        f["audio_f0_max_hz"] = float(np.max(f0v))
        f["audio_f0_median_hz"] = float(np.median(f0v))
        f["audio_f0_range_hz"] = float(np.max(f0v) - np.min(f0v))
        # jitter proxy: mean abs relative change of consecutive F0 values
        adjacent = valid[:-1] & valid[1:]
        changes = np.abs(np.diff(f0)[adjacent]) / f0[:-1][adjacent]
        f["audio_period_variation_proxy"] = float(changes.mean()) if len(changes) else np.nan
        # pitch slope (Hz per second) -> rising/falling intonation
        t = np.flatnonzero(valid) * HOP / SR
        f["audio_f0_slope_hz_per_s"] = float(np.polyfit(t, f0v, 1)[0])
        # semitone variability (speaker-normalised pitch variation)
        st = 12 * np.log2(f0v / np.median(f0v))
        f["audio_f0_std_semitones"] = float(np.std(st))
    # else: leave as NaN (pitch not detected) -> handled in imputation step

    # ---- PROSODIC: energy / loudness ---------------------------------------
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    f["audio_rms_mean"] = float(rms.mean())
    f["audio_rms_std"] = float(rms.std())
    f["audio_rms_max"] = float(rms.max())
    f["audio_rms_min"] = float(rms.min())
    f["audio_rms_frame_variation_proxy"] = float(np.mean(np.abs(np.diff(rms)) / (rms[:-1] + 1e-9)))  # shimmer proxy
    f["audio_rms_db_mean"] = float(np.mean(librosa.amplitude_to_db(rms + 1e-9)))
    f["audio_rms_db_range"] = float(np.ptp(librosa.amplitude_to_db(rms + 1e-9)))

    # ---- PROSODIC: timing / rhythm -----------------------------------------
    active = rms > (0.1 * rms.max())                     # simple energy-based speech activity
    f["audio_active_ratio"] = float(active.mean())       # inverse = pause ratio
    onsets = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP)
    f["audio_onset_rate_per_sec"] = float(len(onsets) / (len(y) / SR))   # speech-rate proxy

    # ---- ACOUSTIC: spectral --------------------------------------------------
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    summ("audio_zcr", zcr, f); f["audio_zcr_max"] = float(zcr.max())

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    cen = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    bw = librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]
    ro = librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0]
    fl = librosa.feature.spectral_flatness(S=S)[0]
    con = librosa.feature.spectral_contrast(S=S, sr=SR)
    summ("audio_spec_centroid", cen, f)
    summ("audio_spec_bandwidth", bw, f)
    summ("audio_spec_rolloff85", ro, f)
    summ("audio_spec_flatness", fl, f)
    summ("audio_spec_contrast", con, f)
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=SR)
    summ("audio_chroma", chroma, f)

    # ---- ACOUSTIC: MFCC + deltas -------------------------------------------
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    width = min(9, (mfcc.shape[1] // 2) * 2 - 1)         # must be odd and <= n_frames
    f["audio_delta_available"] = width >= 3
    f["audio_delta_width"] = width
    if width >= 3:
        d1 = librosa.feature.delta(mfcc, width=width, mode="nearest")
        d2 = librosa.feature.delta(mfcc, order=2, width=width, mode="nearest")
    for i in range(N_MFCC):
        k = i + 1
        summ(f"audio_mfcc_{k}", mfcc[i], f)
        if width >= 3:
            summ(f"audio_delta_mfcc_{k}", d1[i], f)
            summ(f"audio_delta2_mfcc_{k}", d2[i], f)
    return f


def run_extraction(meta):
    rows, errors = [], []
    for i, r in meta.iterrows():
        try:
            feats = extract_features(r["audio_path"])
            feats["participant_code"] = r["participant_code"]
            feats["audio_extraction_ok"] = True
            rows.append(feats)
            print(f"[3] {i+1}/{len(meta)}  {r['participant_code']}  ok")
        except Exception as e:
            errors.append({"participant_code": r["participant_code"], "error": str(e)})
            rows.append({"participant_code": r["participant_code"], "audio_extraction_ok": False})
            print(f"[3] {r['participant_code']} FAILED: {e}")
    pd.DataFrame(errors, columns=["participant_code", "error"]).to_csv(OUT_DIR / "extraction_errors.csv", index=False)
    feat = pd.DataFrame(rows)
    df = meta.drop(columns=["audio_path"]).merge(feat, on="participant_code", how="left")
    df.to_csv(OUT_DIR / "features_raw.csv", index=False)
    if not df["audio_extraction_ok"].all():
        raise RuntimeError("Audio extraction failed; inspect extraction_errors.csv. Raw rows are preserved.")
    print(f"[3] Saved {OUT_DIR/'features_raw.csv'}  shape={df.shape}")
    return df


# ----------------------------------------------------------------------------
# STEP 4 — EXPLORATORY ANALYSIS
# ----------------------------------------------------------------------------
KEY_FEATURES = [
    "audio_f0_mean_hz", "audio_f0_std_hz", "audio_f0_range_hz", "audio_voiced_ratio",
    "audio_rms_mean", "audio_rms_std", "audio_zcr_mean",
    "audio_spec_centroid_mean", "audio_spec_bandwidth_mean", "audio_spec_flatness_mean",
    "audio_active_ratio", "audio_onset_rate_per_sec", "audio_trimmed_dur_sec",
]


def eda(df):
    feat_cols = [c for c in df.columns if c.startswith("audio_") and df[c].dtype != object
                 and c not in ("audio_filename", "audio_sampling_rate", "audio_delta_available",
                               "audio_delta_width", "audio_pitch_total_frames")]
    num = df[feat_cols].astype(float)

    # 4.1 Dataset overview ------------------------------------------------------
    print("\n[4] ===== DATASET OVERVIEW =====")
    print("Recordings:", len(df))
    for c in ["year_level", "class_activity", "subject", "valence_group", "arousal_group", "affect_category"]:
        print(f"\n{c}:\n{df[c].value_counts().to_string()}")
    print(f"\nPitch detected in {df['audio_pitch_detected'].mean():.0%} of recordings "
          f"({(~df['audio_pitch_detected'].astype(bool)).sum()} with no F0 -> NaN)")

    # 4.2 Missing values / descriptive statistics ----------------------------------
    miss = num.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss.to_csv(OUT_DIR / "missing_values.csv", header=["n_missing"])
    desc = num.describe().T
    desc["skew"] = num.skew()
    desc.to_csv(OUT_DIR / "descriptive_stats.csv")
    print("\n[4] Features with missing values (pitch features when unvoiced):\n", miss.head(12).to_string())

    # 4.3 Label distributions ---------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(data=df, x="valence_group", order=["Negative", "Neutral", "Positive"], ax=ax[0])
    sns.countplot(data=df, x="arousal_group", order=["Low", "Moderate", "High"], ax=ax[1])
    sns.heatmap(pd.crosstab(df["valence_group"], df["arousal_group"]), annot=True, fmt="d",
                cmap="Blues", ax=ax[2])
    ax[0].set_title("Valence groups"); ax[1].set_title("Arousal groups"); ax[2].set_title("Valence x Arousal")
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_label_distribution.png", dpi=150); plt.close()

    # 4.4 Distributions of key prosodic/acoustic features ---------------------------
    kf = [c for c in KEY_FEATURES if c in df.columns]
    fig, axes = plt.subplots(3, 5, figsize=(20, 10)); axes = axes.ravel()
    for a, c in zip(axes, kf):
        sns.histplot(df[c].dropna(), kde=True, ax=a); a.set_title(c.replace("audio_", ""), fontsize=9)
    for a in axes[len(kf):]:
        a.axis("off")
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_feature_distributions.png", dpi=150); plt.close()

    # 4.5 Features by emotion group (box plots) ----------------------------------
    for grp, order in [("valence_group", ["Negative", "Neutral", "Positive"]),
                       ("arousal_group", ["Low", "Moderate", "High"])]:
        fig, axes = plt.subplots(3, 5, figsize=(20, 10)); axes = axes.ravel()
        for a, c in zip(axes, kf):
            sns.boxplot(data=df, x=grp, y=c, order=order, ax=a)
            sns.stripplot(data=df, x=grp, y=c, order=order, color="k", size=3, ax=a)
            a.set_title(c.replace("audio_", ""), fontsize=9); a.set_xlabel("")
        for a in axes[len(kf):]:
            a.axis("off")
        plt.tight_layout(); plt.savefig(FIG_DIR / f"03_boxplots_by_{grp}.png", dpi=150); plt.close()

    # 4.6 Correlation with valence / arousal (Spearman — robust for small n) ---------
    rows = []
    for c in feat_cols:
        for target in ["valence_score", "arousal_score"]:
            ok = df[[c, target]].dropna()
            if len(ok) > 5 and ok[c].nunique() > 1:
                rho, p = stats.spearmanr(ok[c], ok[target])
                rows.append({"feature": c, "target": target, "rho": rho, "p_value": p, "n": len(ok)})
    corr = pd.DataFrame(rows).sort_values("p_value")
    from feature_utils import bh_adjust
    corr["q_value_bh"] = bh_adjust(corr["p_value"])
    corr.to_csv(OUT_DIR / "feature_target_correlations.csv", index=False)
    print("\n[4] Top 10 Spearman correlations with valence/arousal:")
    print(corr.head(10).to_string(index=False))

    # 4.7 Group tests (Kruskal-Wallis; non-parametric, fits tiny/unequal groups) ----
    rows = []
    for grp in ["valence_group", "arousal_group", "year_level"]:
        for c in kf:
            groups = [g[c].dropna().values for _, g in df.groupby(grp)]
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                try:
                    h, p = stats.kruskal(*groups)
                    rows.append({"grouping": grp, "feature": c, "H": h, "p_value": p})
                except ValueError:
                    pass
    group_tests = pd.DataFrame(rows, columns=["grouping", "feature", "H", "p_value"])
    group_tests["q_value_bh"] = bh_adjust(group_tests["p_value"])
    group_tests.sort_values("p_value").to_csv(OUT_DIR / "group_tests_kruskal.csv", index=False)

    # 4.8 Correlation heatmap of key features --------------------------------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[kf].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                annot_kws={"size": 7})
    plt.title("Spearman correlation — key prosodic/acoustic features")
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150); plt.close()

    # 4.9 PCA (impute -> scale -> 2 components) -------------------------------------
    X = num.copy()
    X = X.loc[:, X.notna().mean() > 0.5]                       # drop mostly-missing columns
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit(Z)
    pc = pca.transform(Z)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, hue in zip(ax, ["valence_group", "arousal_group"]):
        sns.scatterplot(x=pc[:, 0], y=pc[:, 1], hue=df[hue], style=df["year_level"], s=90, ax=a)
        a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        a.set_title(f"PCA coloured by {hue}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_pca.png", dpi=150); plt.close()

    # 4.10 Outliers (|z| > 3 on key features) ---------------------------------------
    z = (df[kf] - df[kf].mean()) / df[kf].std()
    out = [(df.loc[i, "participant_code"], c, round(z.loc[i, c], 2))
           for c in kf for i in z.index if abs(z.loc[i, c]) > 3]
    pd.DataFrame(out, columns=["participant", "feature", "z"]).to_csv(OUT_DIR / "outliers.csv", index=False)
    print(f"\n[4] Outlier flags (|z|>3): {len(out)}  -> output/outliers.csv")
    return feat_cols


# ----------------------------------------------------------------------------
# STEP 5 — MODEL-READY DATASET
# ----------------------------------------------------------------------------
def make_model_ready(df, feat_cols):
    """Export unimputed features; globally transformed files are explicitly EDA-only."""
    from feature_utils import audio_columns, export_features
    cols = audio_columns(df)
    export_features(df[df["audio_extraction_ok"]].copy(), cols, OUT_DIR, "features")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    unzip_audio()
    meta = load_metadata()
    df = run_extraction(meta)
    feat_cols = eda(df)
    make_model_ready(df, feat_cols)
    print("\nDone. Check the 'output' folder.")