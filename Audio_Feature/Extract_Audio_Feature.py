import glob
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import librosa


# ============================================================
# SETTINGS
# ============================================================

# Your actual folder is "Audios", not "audio"
AUDIO_DIR = "Audios"

# CSV file containing your participant information
LABELS_CSV = "G4_-_RESEARCH_MINI-PROJECT_v2_-_G4_-_2nd_Year.csv"

# Output CSV
OUT_CSV = "audio_feature_dataset.csv"

# Audio settings
SR = 16000
N_FFT = 512
HOP = 160


# ============================================================
# CHECK FILES AND FOLDERS
# ============================================================

if not os.path.exists(AUDIO_DIR):
    raise FileNotFoundError(
        f'Audio folder "{AUDIO_DIR}" was not found. '
        f"Make sure it is inside the same folder as this Python file."
    )

if not os.path.exists(LABELS_CSV):
    raise FileNotFoundError(
        f'CSV file "{LABELS_CSV}" was not found. '
        f"Make sure it is inside the same folder as this Python file."
    )


# ============================================================
# CONVERT AUDIO TO WAV
# ============================================================

def to_wav(src, sr=SR):
    """
    Convert any audio file to mono 16 kHz WAV using FFmpeg.
    """

    wav = os.path.join(
        tempfile.gettempdir(),
        os.path.basename(src) + ".wav"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            src,
            "-ac",
            "1",
            "-ar",
            str(sr),
            wav
        ],
        check=True
    )

    return wav


# ============================================================
# EXTRACT AUDIO FEATURES
# ============================================================

def extract(path):

    # Example:
    # Y2c-001.m4a -> Y2c-001
    code = os.path.splitext(os.path.basename(path))[0]

    print(f"Processing: {os.path.basename(path)}")

    # Convert audio to WAV
    wav_path = to_wav(path)

    # Load audio
    y, sr = librosa.load(wav_path, sr=SR)

    # Remove silence
    yt, _ = librosa.effects.trim(y, top_db=30)

    # --------------------------------------------------------
    # Basic audio information
    # --------------------------------------------------------

    r = {
        "participant_code": code,
        "audio_filename": os.path.basename(path),
        "audio_sr": SR,
        "audio_duration_sec": len(y) / sr,
        "audio_voiced_duration_sec": len(yt) / sr,
    }

    # --------------------------------------------------------
    # Statistics helper
    # --------------------------------------------------------

    def stats(name, a):

        a = np.asarray(a, dtype=float)

        # Remove infinity and NaN values
        a = a[np.isfinite(a)]

        if a.size == 0:
            a = np.array([np.nan])

        r[f"{name}_mean"] = np.nanmean(a)
        r[f"{name}_std"] = np.nanstd(a)
        r[f"{name}_max"] = np.nanmax(a)
        r[f"{name}_min"] = np.nanmin(a)

    # ========================================================
    # ENERGY FEATURES
    # ========================================================

    rms = librosa.feature.rms(
        y=y,
        frame_length=N_FFT,
        hop_length=HOP
    )[0]

    stats("audio_rms", rms)

    r["audio_rms_db_mean"] = float(
        np.mean(
            librosa.amplitude_to_db(
                rms + 1e-9
            )
        )
    )

    # ========================================================
    # SPECTRAL FEATURES
    # ========================================================

    S = np.abs(
        librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP
        )
    )

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=N_FFT,
        hop_length=HOP
    )[0]

    stats("audio_zcr", zcr)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        S=S,
        sr=sr
    )[0]

    stats("audio_spec_centroid", centroid)

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        S=S,
        sr=sr
    )[0]

    stats("audio_spec_bandwidth", bandwidth)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(
        S=S,
        sr=sr
    )[0]

    stats("audio_spec_rolloff", rolloff)

    # Spectral Flatness
    flatness = librosa.feature.spectral_flatness(
        S=S
    )[0]

    stats("audio_spec_flatness", flatness)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(
        S=S,
        sr=sr,
        n_bands=4,
        fmin=100
    ).mean(axis=0)

    stats("audio_spec_contrast", contrast)

    # ========================================================
    # MFCC FEATURES
    # ========================================================

    mf = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=40
    )

    # MFCC 1 to 13
    for i in range(13):

        r[f"audio_mfcc{i + 1}_mean"] = mf[i].mean()
        r[f"audio_mfcc{i + 1}_std"] = mf[i].std()

    # MFCC delta
    if mf.shape[1] >= 5:
        d = librosa.feature.delta(
            mf,
            width=5
        )
    else:
        d = np.zeros_like(mf)

    r["audio_mfcc_delta_mean"] = np.abs(d).mean()

    # ========================================================
    # PITCH FEATURES
    # ========================================================

    f0, vflag, _ = librosa.pyin(
        y,
        fmin=60,
        fmax=500,
        sr=sr,
        frame_length=1024,
        hop_length=HOP
    )

    f0v = f0[~np.isnan(f0)]

    # Voiced ratio
    r["audio_voiced_ratio"] = (
        float(np.mean(vflag))
        if len(vflag)
        else np.nan
    )

    # Pitch statistics
    if len(f0v) >= 2:

        periods = 1 / f0v

        r.update(
            audio_f0_mean=f0v.mean(),
            audio_f0_std=f0v.std(),
            audio_f0_min=f0v.min(),
            audio_f0_max=f0v.max(),
            audio_f0_range=f0v.max() - f0v.min(),
            audio_jitter_local=(
                np.mean(np.abs(np.diff(periods)))
                / np.mean(periods)
            )
        )

    else:

        r.update(
            audio_f0_mean=np.nan,
            audio_f0_std=np.nan,
            audio_f0_min=np.nan,
            audio_f0_max=np.nan,
            audio_f0_range=np.nan,
            audio_jitter_local=np.nan
        )

    # ========================================================
    # SHIMMER
    # ========================================================

    if len(vflag) and len(rms):

        usable_length = min(
            len(rms),
            len(vflag)
        )

        rv = rms[:usable_length][
            vflag[:usable_length]
        ]

    else:

        rv = np.array([])

    if len(rv) >= 2 and np.mean(rv) != 0:

        r["audio_shimmer_local"] = (
            np.mean(np.abs(np.diff(rv)))
            / np.mean(rv)
        )

    else:

        r["audio_shimmer_local"] = np.nan

    # ========================================================
    # HARMONICS-TO-NOISE RATIO
    # ========================================================

    h = librosa.effects.harmonic(y)

    r["audio_hnr_db"] = (
        10
        * np.log10(
            (np.sum(h ** 2) + 1e-9)
            /
            (np.sum((y - h) ** 2) + 1e-9)
        )
    )

    # Delete temporary WAV
    try:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    except Exception:
        pass

    return r


# ============================================================
# BUILD LABEL TEMPLATE
# ============================================================

def build_template(labels_csv):

    """
    Read the research CSV and convert it into
    the dataset template used by the audio features.
    """

    raw = pd.read_csv(labels_csv)

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    raw.columns = (
        raw.columns
        .astype(str)
        .str.replace("**", "", regex=False)
        .str.strip()
    )

    print("\nCSV columns detected:")

    for column in raw.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Create output DataFrame
    # --------------------------------------------------------

    o = pd.DataFrame()

    o["participant_code"] = (
        raw["Participant Code"]
        .astype(str)
        .str.strip()
    )

    o["year_level"] = raw["Year Level"]

    o["class_activity"] = raw["Class Activity"]

    o["subject"] = raw["Subject"]

    o["session"] = raw["Session"].map(
        {
            "Morning": "AM",
            "Afternoon": "PM"
        }
    )

    o["session_name"] = raw["Session"]

    o["spoken_word"] = raw["Spoken Word"]

    o["self_reported_feeling"] = (
        raw["Self-Reported Feeling"]
    )

    # ========================================================
    # VALENCE AND AROUSAL
    # ========================================================

    for col, name in [
        ("Valence (1-5)", "valence"),
        ("Arousal (1-5)", "arousal")
    ]:

        # Convert to string
        values = (
            raw[col]
            .astype(str)
            .str.replace("**", "", regex=False)
            .str.strip()
        )

        # Split:
        # 5 — Very Pleasant
        #
        # into:
        # 5
        # Very Pleasant

        parts = values.str.split(
            " — ",
            n=1,
            expand=True
        )

        # Score
        o[f"{name}_score"] = (
            parts[0]
            .astype(int)
        )

        # Label
        o[f"{name}_label"] = (
            parts[1]
            .str.strip()
        )

    # --------------------------------------------------------
    # Additional fields
    # --------------------------------------------------------

    o["affect_category"] = ""

    o["word_english"] = ""

    o["date_collected"] = raw["Date Collected"]

    return o


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("==========================================")
    print(" AUDIO FEATURE EXTRACTION")
    print("==========================================")

    print(f"\nAudio folder: {AUDIO_DIR}")
    print(f"Labels CSV:   {LABELS_CSV}")
    print(f"Output CSV:   {OUT_CSV}")

    # --------------------------------------------------------
    # Find M4A audio files
    # --------------------------------------------------------

    files = sorted(
        glob.glob(
            os.path.join(
                AUDIO_DIR,
                "*.m4a"
            )
        )
    )

    print(f"\nFound {len(files)} audio files.")

    if len(files) == 0:

        raise FileNotFoundError(
            f'No .m4a files were found inside "{AUDIO_DIR}".\n'
            f"Check that your audio files are inside the Audios folder."
        )

    # --------------------------------------------------------
    # Extract audio features
    # --------------------------------------------------------

    feature_rows = []

    for file in files:

        try:

            feature_rows.append(
                extract(file)
            )

        except Exception as e:

            print(
                f"\nERROR processing {file}:"
            )

            print(e)

    # --------------------------------------------------------
    # Create feature DataFrame
    # --------------------------------------------------------

    if not feature_rows:

        raise RuntimeError(
            "No audio features were successfully extracted."
        )

    feats = pd.DataFrame(feature_rows)

    print(
        f"\nSuccessfully extracted features from "
        f"{len(feats)} audio files."
    )

    print("\nFeature columns:")
    print(
        feats.columns.tolist()
    )

    # --------------------------------------------------------
    # Build research template
    # --------------------------------------------------------

    template = build_template(
        LABELS_CSV
    )

    print(
        f"\nParticipants in CSV: "
        f"{len(template)}"
    )

    # --------------------------------------------------------
    # Check participant codes
    # --------------------------------------------------------

    print("\nParticipant codes from CSV:")

    print(
        template["participant_code"]
        .tolist()
    )

    print("\nParticipant codes from audio:")

    print(
        feats["participant_code"]
        .tolist()
    )

    # --------------------------------------------------------
    # Merge labels + audio features
    # --------------------------------------------------------

    out = template.merge(
        feats,
        on="participant_code",
        how="left",
        validate="1:1"
    )

    # --------------------------------------------------------
    # Check missing audio
    # --------------------------------------------------------

    missing = (
        out["audio_filename"]
        .isna()
        .sum()
    )

    if missing:

        print(
            f"\nWARNING: {missing} participants "
            f"have no matching audio file."
        )

        missing_codes = (
            out.loc[
                out["audio_filename"].isna(),
                "participant_code"
            ]
            .tolist()
        )

        print(
            "Missing audio for:"
        )

        for code in missing_codes:
            print(
                f"  - {code}"
            )

    else:

        print(
            "\nAll participants have matching audio files."
        )

    # --------------------------------------------------------
    # Save final dataset
    # --------------------------------------------------------

    out.round(5).to_csv(
        OUT_CSV,
        index=False
    )

    print("\n==========================================")

    print(
        f"Saved {OUT_CSV}"
    )

    print(
        f"Rows:    {out.shape[0]}"
    )

    print(
        f"Columns: {out.shape[1]}"
    )

    print("==========================================")