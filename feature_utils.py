"""Explicit predictor selection and exports. Never fit training preprocessing here."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

IDS = ["participant_code", "year_level", "valence_score", "arousal_score",
       "valence_group", "arousal_group", "affect_category"]
AUDIO_QC = {"audio_extraction_ok", "audio_pitch_detected", "audio_sampling_rate",
            "audio_original_sampling_rate", "audio_original_channels", "audio_clipping_ratio",
            "audio_delta_available", "audio_delta_width", "audio_pitch_total_frames",
            "audio_voiced_frames", "audio_duration_sec", "audio_trimmed_fraction"}
FACE_QC = {"face_detected_frames", "face_detection_rate", "face_detection_quality_pass"}

def audio_columns(df):
    return [c for c in df if c.startswith("audio_") and c not in AUDIO_QC
            and pd.api.types.is_numeric_dtype(df[c])]

def facial_columns(df):
    # These composites are aliases of the original blendshapes, not new information.
    aliases = ("face_brow_inner_raiser_", "face_jaw_open_", "face__neutral")
    return [c for c in df if c.startswith("face_") and c not in FACE_QC
            and not c.startswith(aliases) and pd.api.types.is_numeric_dtype(df[c])]

def export_features(df, cols, out, prefix):
    out.mkdir(parents=True, exist_ok=True)
    if df.empty or not cols:
        raise ValueError("No usable samples/features to export")
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    pd.concat([df[IDS], X], axis=1).to_csv(out / f"{prefix}_unimputed.csv", index=False)
    # Global fitting below is for visual EDA only, never for held-out evaluation.
    E = X.loc[:, X.notna().mean() > .5].copy()
    E = E.fillna(E.median())
    E = E.loc[:, E.std(ddof=0) > 0]
    if E.empty:
        raise ValueError("No varying EDA predictors")
    Z = pd.DataFrame(StandardScaler().fit_transform(E), columns=E.columns, index=E.index)
    pd.concat([df[IDS], E], axis=1).to_csv(out / f"{prefix}_eda_imputed.csv", index=False)
    pd.concat([df[IDS], Z], axis=1).to_csv(out / f"{prefix}_eda_scaled.csv", index=False)
    # Remove legacy names so old leakage-prone exports cannot be mistaken for new data.
    for suffix in ("clean_unscaled", "model_ready_scaled"):
        (out / f"{prefix}_{suffix}.csv").unlink(missing_ok=True)
    print(f"Exported {len(df)} samples, {len(cols)} raw predictors; EDA-only transformed copies")


def bh_adjust(p):
    """Benjamini-Hochberg correction, retaining missing p-values."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        ranked = p[order] * len(order) / np.arange(1, len(order)+1)
        q[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return q

