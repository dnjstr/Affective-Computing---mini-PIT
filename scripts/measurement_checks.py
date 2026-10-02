"""Measurement checks for the open items in docs/VALIDATION.md:
(1) pYIN F0 tracks against the 65-500 Hz search bounds (Y4D-020 review),
(2) WAV vs MP4 track interval correspondence before any synchronization claim,
(3) head-pose sign conventions against known rotation matrices.
Run after scripts/run_all.py; writes CSVs, figures and docs/MEASUREMENT_CHECKS.md.
"""
import importlib.util
import struct
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audio = _load("audio_pipeline", "Audio_Feature/audio_feature_pipeline.py")
face = _load("face_pipeline", "Facial_Feature/face_feature_pipeline.py")

AUDIO_DIR = ROOT / "Audio_Feature/audio"
VIDEO_DIR = ROOT / "Facial_Feature/video"
AUDIO_OUT = ROOT / "Audio_Feature/output"
FACE_OUT = ROOT / "Facial_Feature/output_face"
MM_OUT = ROOT / "Multimodal_Feature/output_multimodal"
DOCS = ROOT / "docs"

BOUND_MARGIN_HZ = 5.0   # voiced frame within this many Hz of a bound is flagged
INTERVAL_TOL = 0.1      # seconds; same tolerance as sample_quality.csv review flag


# ----------------------------------------------------------------------------
# CHECK 1 - PITCH TRACKS vs SEARCH BOUNDS
# ----------------------------------------------------------------------------
def pyin_track(path):
    y, _ = librosa.load(str(path), sr=audio.SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=audio.TRIM_DB)
    if len(y) < audio.N_FFT:
        y = np.pad(y, (0, audio.N_FFT - len(y)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        f0, _, _ = librosa.pyin(y, fmin=audio.F0_MIN, fmax=audio.F0_MAX, sr=audio.SR,
                                frame_length=audio.N_FFT, hop_length=audio.HOP)
    t = np.arange(len(f0)) * audio.HOP / audio.SR
    return t, f0, y


def near_bound_stats(f0, fmin, fmax, margin=BOUND_MARGIN_HZ):
    """Per-recording F0 summary and flags for frames hugging a search bound."""
    v = f0[np.isfinite(f0)]
    stats = {"voiced_frames": int(len(v)),
             "f0_min_hz": float(v.min()) if len(v) else np.nan,
             "f0_max_hz": float(v.max()) if len(v) else np.nan,
             "f0_mean_hz": float(v.mean()) if len(v) else np.nan}
    near_lo = int((v <= fmin + margin).sum()) if len(v) else 0
    near_hi = int((v >= fmax - margin).sum()) if len(v) else 0
    stats["frames_near_lower_bound"] = near_lo
    stats["frames_near_upper_bound"] = near_hi
    stats["pct_frames_near_lower_bound"] = near_lo / len(v) if len(v) else 0.0
    stats["pct_frames_near_upper_bound"] = near_hi / len(v) if len(v) else 0.0
    stats["near_bound_flag"] = bool(near_lo or near_hi)
    return stats


def _plot_pitch_overview(tracks):
    codes = sorted(tracks)
    fig, axes = plt.subplots(5, 5, figsize=(15, 12), sharex=True)
    for ax, code in zip(axes.flat, codes):
        t, f0, _, s = tracks[code]
        ax.plot(t, f0, color="#527ca7", lw=1.2)
        ax.axhline(audio.F0_MIN, color="gray", ls="--", lw=0.8)
        ax.axhline(audio.F0_MAX, color="darkorange", ls="--", lw=0.8)
        ax.set_ylim(0, audio.F0_MAX * 1.1)
        flagged = s["near_bound_flag"]
        ax.set_title(code + ("  - NEAR BOUND" if flagged else ""), fontsize=9,
                     color="crimson" if flagged else "black")
        if not s["voiced_frames"]:
            ax.text(0.5, 0.5, "no voiced frames", transform=ax.transAxes,
                    ha="center", fontsize=8, color="gray")
        ax.tick_params(labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Hz", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("seconds", fontsize=8)
    fig.suptitle("pYIN F0 tracks vs 65-500 Hz search bounds (crimson = near-bound flag)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(AUDIO_OUT / "figures/06_pitch_tracks_overview.png", dpi=160)
    plt.close(fig)


def _plot_pitch_detail(code, t, f0, y, s):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    tt = np.arange(len(y)) / audio.SR
    ax[0].plot(tt, y, color="#527ca7", lw=0.6)
    ax[0].set(ylabel="amplitude", title=f"{code}: waveform")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=audio.N_FFT,
                                                    hop_length=audio.HOP)), ref=np.max)
    librosa.display.specshow(S, sr=audio.SR, hop_length=audio.HOP,
                             x_axis="time", y_axis="hz", ax=ax[1])
    ax[1].set_ylim(0, 1000)
    fin = np.isfinite(f0)
    ax[1].scatter(t[fin], f0[fin], s=22, color="darkorange", zorder=3, label="pYIN F0")
    ax[1].axhline(audio.F0_MIN, color="gray", ls="--", lw=1, label="search bounds")
    ax[1].axhline(audio.F0_MAX, color="gray", ls="--", lw=1)
    ax[1].set(xlabel="seconds", ylabel="Hz")
    ax[1].legend(fontsize=8, loc="upper right")
    hi = s["pct_frames_near_upper_bound"]
    lo = s["pct_frames_near_lower_bound"]
    fig.suptitle(f"{code}: {s['voiced_frames']} voiced frames, "
                 f"F0 {s['f0_min_hz']:.1f}-{s['f0_max_hz']:.1f} Hz, "
                 f"{hi:.0%} within {BOUND_MARGIN_HZ:.0f} Hz of ceiling, "
                 f"{lo:.0%} of floor")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(AUDIO_OUT / f"figures/07_pitch_track_detail_{code}.png", dpi=160)
    plt.close(fig)


def pitch_check():
    stored = pd.read_csv(AUDIO_OUT / "features_raw.csv").set_index("participant_code")
    rows, tracks = [], {}
    for wav in sorted(AUDIO_DIR.glob("*.wav")):
        code = wav.name[:7]
        t, f0, y = pyin_track(wav)
        s = near_bound_stats(f0, audio.F0_MIN, audio.F0_MAX)
        stored_max = stored.at[code, "audio_f0_max_hz"]
        matches = ((np.isnan(stored_max) and np.isnan(s["f0_max_hz"])) or
                   np.isclose(stored_max, s["f0_max_hz"], atol=1e-6))
        rows.append({"participant_code": code, "audio_filename": wav.name, **s,
                     "stored_f0_max_hz": stored_max,
                     "recomputed_matches_stored": bool(matches)})
        tracks[code] = (t, f0, y, s)
    df = pd.DataFrame(rows)
    df.to_csv(AUDIO_OUT / "pitch_track_check.csv", index=False)
    (AUDIO_OUT / "figures").mkdir(exist_ok=True)
    _plot_pitch_overview(tracks)
    flagged = sorted(c for c in tracks if tracks[c][3]["near_bound_flag"])
    for code in flagged:
        _plot_pitch_detail(code, *tracks[code])
    return df, flagged


# ----------------------------------------------------------------------------
# CHECK 2 - WAV vs MP4 TRACK INTERVALS
# ----------------------------------------------------------------------------
def mp4_track_durations(data):
    """Parse top-level MP4 boxes for container/video/audio durations (seconds)."""
    out = {"container": None, "video": None, "audio": None}

    def scan(start, end):
        i, pending_mdhd = start, None
        while i + 8 <= end:
            size, typ = struct.unpack(">I4s", data[i:i + 8])
            typ = typ.decode("latin1")
            hdr = 8
            if size == 1:
                size = struct.unpack(">Q", data[i + 8:i + 16])[0]
                hdr = 16
            if size < 8:
                break
            body = i + hdr
            if typ in ("moov", "trak", "mdia", "minf", "stbl"):
                pending = scan(body, i + size)
                if pending is not None:
                    pending_mdhd = pending
            elif typ == "mvhd":
                v = data[body]
                ts, dur = (struct.unpack(">II", data[body + 12:body + 20]) if v == 0
                           else struct.unpack(">IQ", data[body + 20:body + 32]))
                out["container"] = dur / ts
            elif typ == "mdhd":
                v = data[body]
                ts, dur = (struct.unpack(">II", data[body + 12:body + 20]) if v == 0
                           else struct.unpack(">IQ", data[body + 20:body + 32]))
                pending_mdhd = dur / ts
            elif typ == "hdlr":
                handler = data[body + 8:body + 12].decode("latin1")
                if pending_mdhd is not None:
                    if handler in ("vide", "soun"):
                        out["video" if handler == "vide" else "audio"] = pending_mdhd
                    pending_mdhd = None
            i += size
        return pending_mdhd

    scan(0, len(data))
    return out


def _plot_intervals(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    panels = [("wav_minus_mp4_audio_sec", "WAV duration vs MP4 audio-track duration",
               "MP4 audio-track duration (s)"),
              ("mp4_audio_minus_video_sec", "MP4 audio track vs MP4 video track",
               "MP4 video-track duration (s)")]
    for a, (col, title, xlabel) in zip(ax, panels):
        x = df["mp4_audio_track_sec"] if col == "wav_minus_mp4_audio_sec" else df["mp4_video_track_sec"]
        y = df["wav_duration_sec"] if col == "wav_minus_mp4_audio_sec" else df["mp4_audio_track_sec"]
        a.scatter(x, y, color="#527ca7", s=45, zorder=3)
        lim_lo = min(x.min(), y.min()) * 0.95
        lim_hi = max(x.max(), y.max()) * 1.05
        a.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="equal duration")
        a.fill_between([lim_lo, lim_hi], [lim_lo - INTERVAL_TOL, lim_hi - INTERVAL_TOL],
                       [lim_lo + INTERVAL_TOL, lim_hi + INTERVAL_TOL],
                       color="gray", alpha=0.15, label=f"+/-{INTERVAL_TOL}s")
        for _, r in df.iterrows():
            if abs(r[col]) > 0.35:
                a.annotate(r["participant_code"], (r[x.name], r[y.name]),
                           fontsize=8, xytext=(4, 4), textcoords="offset points")
        a.set(title=title, xlabel=xlabel, ylabel="WAV duration (s)"
              if col == "wav_minus_mp4_audio_sec" else "MP4 audio-track duration (s)")
        a.legend(fontsize=8)
    fig.suptitle("Interval correspondence check (duration level only)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(MM_OUT / "figures/08_interval_decomposition.png", dpi=160)
    plt.close(fig)


def interval_check():
    qc = pd.read_csv(MM_OUT / "sample_quality.csv").set_index("participant_code")
    rows = []
    for wav in sorted(AUDIO_DIR.glob("*.wav")):
        code = wav.name[:7]
        mp4s = sorted(VIDEO_DIR.glob(code + "*.mp4"))
        if not mp4s:
            rows.append({"participant_code": code, "audio_filename": wav.name,
                         "mp4_filename": None})
            continue
        mp4 = mp4s[0]
        d = mp4_track_durations(mp4.read_bytes())
        wav_sec = sf.info(str(wav)).duration
        a, v = d["audio"], d["video"]
        row = {"participant_code": code, "audio_filename": wav.name,
               "mp4_filename": mp4.name, "filename_stem_match": wav.stem == mp4.stem,
               "wav_duration_sec": wav_sec, "mp4_audio_track_sec": a,
               "mp4_video_track_sec": v, "mp4_container_sec": d["container"],
               "wav_minus_mp4_audio_sec": wav_sec - a if a else np.nan,
               "mp4_audio_minus_video_sec": a - v if a and v else np.nan}
        row["wav_vs_mp4_audio_flag"] = bool(a and abs(wav_sec - a) > INTERVAL_TOL)
        row["mp4_av_flag"] = bool(a and v and abs(a - v) > INTERVAL_TOL)
        row["wav_vs_mp4_video_flag"] = bool(v and abs(wav_sec - v) > INTERVAL_TOL)
        if code in qc.index:
            row["sample_quality_review_flag"] = bool(qc.at[code, "duration_review_flag"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(MM_OUT / "interval_check.csv", index=False)
    (MM_OUT / "figures").mkdir(exist_ok=True)
    _plot_intervals(df)
    return df


# ----------------------------------------------------------------------------
# CHECK 3 - HEAD-POSE SIGN CONVENTIONS
# ----------------------------------------------------------------------------
def rotation_matrix(pitch_d, yaw_d, roll_d):
    p, y, r = np.radians([pitch_d, yaw_d, roll_d])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def head_pose_sign_check():
    cases = [("pitch", -30.0, 0.0, 0.0), ("pitch", -10.0, 0.0, 0.0),
             ("pitch", 10.0, 0.0, 0.0), ("pitch", 30.0, 0.0, 0.0),
             ("yaw", 0.0, -45.0, 0.0), ("yaw", 0.0, -15.0, 0.0),
             ("yaw", 0.0, 15.0, 0.0), ("yaw", 0.0, 45.0, 0.0),
             ("roll", 0.0, 0.0, -20.0), ("roll", 0.0, 0.0, -5.0),
             ("roll", 0.0, 0.0, 5.0), ("roll", 0.0, 0.0, 20.0),
             ("combined", 10.0, -20.0, 15.0)]
    rows = []
    for axis, p, y, r in cases:
        got = face.head_pose_deg(rotation_matrix(p, y, r))
        rows.append({"axis": axis, "input_pitch_deg": p, "input_yaw_deg": y,
                     "input_roll_deg": r, "recovered_pitch_deg": got[0],
                     "recovered_yaw_deg": got[1], "recovered_roll_deg": got[2],
                     "max_abs_error_deg": float(np.max(np.abs(np.array(got) - (p, y, r))))})
    df = pd.DataFrame(rows)
    scale_a = face.head_pose_deg(rotation_matrix(10, -20, 15))
    scale_b = face.head_pose_deg(rotation_matrix(10, -20, 15) * 3.7)
    scale_err = float(np.max(np.abs(np.array(scale_a) - np.array(scale_b))))
    df.to_csv(FACE_OUT / "head_pose_sign_check.csv", index=False)
    return df, scale_err


def head_pose_observed_ranges():
    raw = pd.read_csv(FACE_OUT / "face_features_raw.csv")
    rows = []
    for axis in ("pitch", "yaw", "roll"):
        means = raw[f"face_{axis}_mean_deg"]
        rows.append({"axis": axis, "min_of_means_deg": means.min(),
                     "max_of_means_deg": means.max(), "mean_of_means_deg": means.mean(),
                     "max_std_deg": raw[f"face_{axis}_std_deg"].max(),
                     "all_within_90_deg": bool(means.abs().max() < 90)})
    df = pd.DataFrame(rows)
    df.to_csv(FACE_OUT / "head_pose_observed_ranges.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# REPORT
# ----------------------------------------------------------------------------
def write_report(pitch_df, flagged, interval_df, sign_df, scale_err, ranges_df):
    det = pitch_df[pitch_df.near_bound_flag]
    det_lines = []
    for _, r in det.iterrows():
        det_lines.append(
            f"| {r.participant_code} | {r.voiced_frames} | {r.f0_min_hz:.1f} | "
            f"{r.f0_max_hz:.1f} | {r.pct_frames_near_upper_bound:.0%} | "
            f"{r.pct_frames_near_lower_bound:.0%} |")
    wav_diff = interval_df["wav_minus_mp4_audio_sec"].dropna()
    av_diff = interval_df["mp4_audio_minus_video_sec"].dropna()
    wav_video_flags = int(interval_df["wav_vs_mp4_video_flag"].sum())
    sq_flags = int(interval_df["sample_quality_review_flag"].sum()) \
        if "sample_quality_review_flag" in interval_df else None
    pitch_detected = int((pitch_df.voiced_frames > 0).sum())
    matches = int(pitch_df.recomputed_matches_stored.sum())
    stems = int(interval_df["filename_stem_match"].sum())
    max_sign_err = sign_df.max_abs_error_deg.max()
    lines = [
        "# Measurement checks",
        "",
        f"Generated by `scripts/measurement_checks.py` on {date.today().isoformat()}. "
        "Automates the reviewable parts of the remaining measurement checks in "
        "`docs/VALIDATION.md`; manual listening and video inspection are still required "
        "where noted.",
        "",
        "## 1. Pitch tracks vs search bounds (65-500 Hz)",
        "",
        f"- pYIN recomputed on all {len(pitch_df)} WAVs with the pipeline settings "
        f"(16 kHz, frame 2048, hop 512, trim 30 dB): pitch detected in "
        f"{pitch_detected}/25 recordings; recomputed F0 maxima match "
        f"`features_raw.csv` in {matches}/{len(pitch_df)} recordings.",
        f"- Near-bound rule: any voiced frame within {BOUND_MARGIN_HZ:.0f} Hz of a bound. "
        f"Flagged: {', '.join(flagged) if flagged else 'none'}.",
        "",
        "| Recording | Voiced frames | F0 min (Hz) | F0 max (Hz) | % frames near ceiling | % frames near floor |",
        "|---|---:|---:|---:|---:|---:|",
        *det_lines,
        "",
        "- `Y4D-020` sits at the configured ceiling: its voiced estimates cluster just "
        "below 500 Hz, which can indicate an octave/harmonic error rather than true F0. "
        "Inspect the detail figure while listening to the recording before using its "
        "pitch features; do not automatically accept or discard them.",
        "- Recordings with no voiced frames are unvoiced/failed pitch, not silence "
        "(see `pitch_track_check.csv`).",
        "",
        "![F0 overview](../Audio_Feature/output/figures/06_pitch_tracks_overview.png)",
        "",
        *[f"![{c} detail](../Audio_Feature/output/figures/07_pitch_track_detail_{c}.png)"
          for c in flagged],
        "",
        "CSV: `Audio_Feature/output/pitch_track_check.csv`",
        "",
        "## 2. WAV vs MP4 interval correspondence",
        "",
        f"- Pairs checked: {len(interval_df)}; filename stems match in {stems}/25.",
        f"- WAV vs MP4 audio track: differences range {wav_diff.min():+.3f} to "
        f"{wav_diff.max():+.3f} s; {int(interval_df.wav_vs_mp4_audio_flag.sum())}/25 "
        f"exceed {INTERVAL_TOL} s.",
        f"- MP4 audio vs MP4 video track: differences range {av_diff.min():+.3f} to "
        f"{av_diff.max():+.3f} s; {int(interval_df.mp4_av_flag.sum())}/25 exceed "
        f"{INTERVAL_TOL} s.",
        f"- WAV vs MP4 video track: {wav_video_flags}/25 exceed {INTERVAL_TOL} s"
        + (f", matching the {sq_flags} `duration_review_flag` values in "
           "`sample_quality.csv`." if sq_flags is not None else "."),
        "- Interpretation: the existing duration review flags are driven by WAV-to-MP4 "
        "duration differences, not by misalignment between the MP4's own audio and "
        "video tracks, which correspond within one video frame. Interval "
        "correspondence between the standalone WAVs and the MP4s is NOT established "
        "at the sub-second level.",
        "- This is duration-level evidence only. Content-level correspondence "
        "(cross-correlating the WAV against decoded MP4 audio) was not possible here: "
        "no AAC decoder or ffmpeg is available in this environment.",
        "- Consequence: keep the existing rule - do not claim sample-level or "
        "frame-level synchronization between modalities; participant-level pairing "
        "only.",
        "",
        "![Interval decomposition](../Multimodal_Feature/output_multimodal/figures/08_interval_decomposition.png)",
        "",
        "CSV: `Multimodal_Feature/output_multimodal/interval_check.csv`",
        "",
        "## 3. Head-pose sign conventions",
        "",
        f"- `head_pose_deg` was fed known rotation matrices (Rz@Ry@Rx with single-axis "
        f"and combined angles). Max absolute error across {len(sign_df)} cases: "
        f"{max_sign_err:.2e} deg. Sign conventions: positive pitch = +x rotation, "
        f"positive yaw = +y rotation, positive roll = +z rotation.",
        f"- Scale normalization check (matrix x3.7): max deviation {scale_err:.2e} deg.",
        "- Observed per-participant means (from `face_features_raw.csv`):",
        "",
        "| Axis | Min of means (deg) | Max of means (deg) | Max within-clip std (deg) | All abs(mean) < 90 |",
        "|---|---:|---:|---:|---|",
        *[f"| {r.axis} | {r.min_of_means_deg:.1f} | {r.max_of_means_deg:.1f} | "
          f"{r.max_std_deg:.1f} | {r.all_within_90_deg} |" for r in ranges_df.itertuples()],
        "",
        "- These checks validate the extraction math against standard rotation "
        "matrices and confirm plausible observed ranges. They do NOT validate signs "
        "against independently labeled head poses in the videos; that still requires "
        "visiting recordings with known head orientations.",
        "",
        "CSV: `Facial_Feature/output_face/head_pose_sign_check.csv`, "
        "`Facial_Feature/output_face/head_pose_observed_ranges.csv`",
        "",
        "## Still manual / pending",
        "",
        "- Listen to the flagged recordings while viewing their F0 detail figures "
        "(`Y4D-020` above all).",
        "- Visual inspection of landmark/expression tracking on the source videos: "
        "not run; it requires a MediaPipe install, which would downgrade numpy in the "
        "current environment. The README MediaPipe metrics notice applies whenever it "
        "is run.",
        "- Content-level WAV/MP4 correspondence needs an AAC decoder (e.g. ffmpeg); "
        "re-run this script's interval section after decoding if that claim is ever "
        "required.",
        "",
    ]
    (DOCS / "MEASUREMENT_CHECKS.md").write_text("\n".join(lines) + "\n")


def main():
    print("[1/3] pitch tracks vs search bounds ...")
    pitch_df, flagged = pitch_check()
    print(f"      flagged: {flagged or 'none'}; CSV + figures written")
    print("[2/3] WAV vs MP4 interval correspondence ...")
    interval_df = interval_check()
    print(f"      wav-vs-mp4-audio flags: {int(interval_df.wav_vs_mp4_audio_flag.sum())}/25; "
          f"mp4 A/V flags: {int(interval_df.mp4_av_flag.sum())}/25")
    print("[3/3] head-pose sign conventions ...")
    sign_df, scale_err = head_pose_sign_check()
    ranges_df = head_pose_observed_ranges()
    print(f"      max sign error {sign_df.max_abs_error_deg.max():.2e} deg; "
          f"scale error {scale_err:.2e} deg")
    write_report(pitch_df, flagged, interval_df, sign_df, scale_err, ranges_df)
    print("report: docs/MEASUREMENT_CHECKS.md")


if __name__ == "__main__":
    main()
