import re
import math
import zipfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "G4_-_RESEARCH_MINI-PROJECT.csv"
VIDEO_DIR = BASE / "video"                 
ZIP_PATH = None                             
OUT_DIR = BASE / "output_face"
FIG_DIR = OUT_DIR / "figures"
MODEL_PATH = BASE / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

MIN_DETECTION_RATE = 0.80                

for d in (OUT_DIR, FIG_DIR, VIDEO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# MediaPipe 468-point mesh indices
R_EYE = [33, 160, 158, 133, 153, 144]       
L_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_UP, MOUTH_LOW, MOUTH_L, MOUTH_R = 13, 14, 78, 308   # inner lips + corners


# ----------------------------------------------------------------------------
# STEP 1 - METADATA
# ----------------------------------------------------------------------------
def unzip_videos():
    zips = [Path(ZIP_PATH)] if ZIP_PATH else sorted(BASE.glob("*.zip"))
    for z_path in zips:
        with zipfile.ZipFile(z_path) as z:
            names = z.namelist()
            if any(n.lower().endswith((".mp4", ".mov")) for n in names):
                z.extractall(VIDEO_DIR)
                print(f"[1] Extracted {z_path.name} -> {VIDEO_DIR}")


def _num(s):
    m = re.match(r"\s*(\d)", str(s))
    return int(m.group(1)) if m else np.nan


VAL_LABEL = {1: "Very Unpleasant", 2: "Unpleasant", 3: "Neutral", 4: "Pleasant", 5: "Very Pleasant"}
ARO_LABEL = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
group_valence = lambda v: "Negative" if v <= 2 else ("Neutral" if v == 3 else "Positive")
group_arousal = lambda a: "Low" if a <= 2 else ("Moderate" if a == 3 else "High")


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
        "spoken_word": df["Spoken Word"].str.strip(),            # reference only, NOT a feature
        "self_reported_feeling": df["Self-Reported Feeling"].str.strip(),
        "valence_score": df["Valence (1-5)"].map(_num),
        "arousal_score": df["Arousal (1-5)"].map(_num),
        "date_collected": df["Date Collected"],
        "consent_obtained": df["Consent Obtained"].str.strip().str.lower().eq("yes"),
        "video_filename": df["Video Filename"].str.strip(),
    })
    out["valence_label"] = out["valence_score"].map(VAL_LABEL)
    out["arousal_label"] = out["arousal_score"].map(ARO_LABEL)
    out["valence_group"] = out["valence_score"].apply(group_valence)
    out["arousal_group"] = out["arousal_score"].apply(group_arousal)
    out["affect_category"] = out["valence_group"] + "-" + out["arousal_group"]

    assert out["participant_code"].is_unique, "Duplicate participant codes!"
    if (~out["consent_obtained"]).any():
        print(f"[1] WARNING: {(~out['consent_obtained']).sum()} rows without consent -> excluded")
        out = out[out["consent_obtained"]]

    lookup = {p.name.lower(): p for p in BASE.rglob("*")
              if p.suffix.lower() in (".mp4", ".mov") and OUT_DIR not in p.parents}
    print(f"[1] Found {len(lookup)} video files under {BASE}")
    if not lookup:
        raise SystemExit("\nNo video files found. Put the video .zip (or the .mp4 files) in the same "
                         f"folder as this script:\n  {BASE}\nthen run again.")
    out["video_path"] = out["video_filename"].str.lower().map(lookup)
    missing = out[out["video_path"].isna()]
    if len(missing):
        print("[1] WARNING: video not found for:", missing["participant_code"].tolist())
        out = out[out["video_path"].notna()]
    print(f"[1] Metadata ready: {len(out)} usable recordings")
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# STEP 2 - MEDIAPIPE SETUP
# ----------------------------------------------------------------------------
def ensure_model():
    if MODEL_PATH.exists():
        return
    print(f"[2] Downloading face_landmarker.task ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise SystemExit(
            f"\nCould not download the model automatically ({e}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {MODEL_PATH}\nthen run again.")


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    ensure_model()
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)


# ----------------------------------------------------------------------------
# STEP 3 - PER-FRAME SIGNALS
# ----------------------------------------------------------------------------
def _d(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts, idx):
    p = [pts[i] for i in idx]
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]) + 1e-9)


def mouth_aspect_ratio(pts):
    return _d(pts[MOUTH_UP], pts[MOUTH_LOW]) / (_d(pts[MOUTH_L], pts[MOUTH_R]) + 1e-9)


def head_pose_deg(matrix):
    """pitch / yaw / roll in degrees from the 4x4 facial transformation matrix."""
    R = np.array(matrix, dtype=float)[:3, :3]
    R = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-9)     # remove scale
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def frame_signals(result, w, h):
    """Return dict of signals for one frame, or None if no face was found."""
    if not result.face_landmarks:
        return None
    s = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
    pts = np.array([[l.x * w, l.y * h] for l in result.face_landmarks[0]])   # 2D pixel coords
    s["_ear"] = (eye_aspect_ratio(pts, L_EYE) + eye_aspect_ratio(pts, R_EYE)) / 2.0
    s["_mar"] = mouth_aspect_ratio(pts)
    if result.facial_transformation_matrixes is not None and len(result.facial_transformation_matrixes):
        s["_pitch"], s["_yaw"], s["_roll"] = head_pose_deg(result.facial_transformation_matrixes[0])
    return s


# ----------------------------------------------------------------------------
# STEP 3b - PER-VIDEO FEATURES
# ----------------------------------------------------------------------------
def aggregate(frames, n_decoded, fps, w, h, n_reported, brightness, sharpness):
    """frames: list of per-frame dicts (None entries = face not detected)."""
    f = {
        "video_fps": fps,
        "video_reported_frames": n_reported,
        "video_decoded_frames": n_decoded,
        "video_duration_sec": round(n_decoded / fps, 4) if fps else np.nan,
        "video_width_px": w, "video_height_px": h,
        "video_brightness_mean": brightness,        # QC: lighting
        "video_sharpness_mean": sharpness,          # QC: blur (Laplacian variance)
        "face_geometry_coordinate_system": "2D pixel coordinates",
    }
    good = [x for x in frames if x is not None]
    f["face_detected_frames"] = len(good)
    f["face_detection_rate"] = round(len(good) / max(n_decoded, 1), 4)
    f["face_detection_quality_pass"] = f["face_detection_rate"] >= MIN_DETECTION_RATE
    if not good:
        return f

    df = pd.DataFrame(good)
    blend_cols = [c for c in df.columns if not c.startswith("_") or c == "_neutral"]
    # --- 52 blendshapes: mean / std / max --------------------------------------------
    for c in blend_cols:
        f[f"face_{c}_mean"] = df[c].mean()
        f[f"face_{c}_std"] = df[c].std(ddof=0)
        f[f"face_{c}_max"] = df[c].max()

    g = lambda *names: df[list(names)].mean(axis=1)      

    # --- interpretable action-unit style composites ------------------------------------
    comps = {
        "smile": g("mouthSmileLeft", "mouthSmileRight"),
        "frown": g("mouthFrownLeft", "mouthFrownRight"),
        "brow_lowerer": g("browDownLeft", "browDownRight"),
        "brow_inner_raiser": df["browInnerUp"],
        "jaw_open": df["jawOpen"],
    }
    for k, v in comps.items():
        f[f"face_{k}_mean"], f[f"face_{k}_std"], f[f"face_{k}_max"] = v.mean(), v.std(ddof=0), v.max()
    f["face_eye_squint_mean"] = g("eyeSquintLeft", "eyeSquintRight").mean()
    f["face_disgust_proxy_mean"] = g("noseSneerLeft", "noseSneerRight",
                                     "mouthUpperUpLeft", "mouthUpperUpRight").mean()

    # --- expressiveness & motion ------------------------------------------------------
    real = [c for c in blend_cols if c != "_neutral"]
    f["face_expressiveness_proxy"] = df[real].std(ddof=0).mean()        # avg temporal variability
    if len(df) > 1:
        f["face_motion_energy"] = df[real].diff().abs().mean().mean()   # avg frame-to-frame change

    # --- geometry ---------------------------------------------------------------------
    f["face_ear_mean"], f["face_ear_std"], f["face_ear_min"] = df["_ear"].mean(), df["_ear"].std(ddof=0), df["_ear"].min()
    f["face_mar_mean"], f["face_mar_std"], f["face_mar_max"] = df["_mar"].mean(), df["_mar"].std(ddof=0), df["_mar"].max()
    for a in ("pitch", "yaw", "roll"):
        if f"_{a}" in df:
            f[f"face_{a}_mean_deg"], f[f"face_{a}_std_deg"] = df[f"_{a}"].mean(), df[f"_{a}"].std(ddof=0)
    return f


def process_video(path, landmarker):
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, bright, sharp, i = [], [], [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bright.append(gray.mean())
        sharp.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
        frames.append(frame_signals(res, w, h))
        i += 1
    cap.release()
    return aggregate(frames, i, fps, w, h, n_reported, float(np.mean(bright)) if bright else np.nan,
                     float(np.mean(sharp)) if sharp else np.nan)


def run_extraction(meta, landmarker_factory=None):
    """A FRESH landmarker is created for every video: in VIDEO mode MediaPipe requires timestamps
    to keep increasing for the lifetime of one landmarker, so re-using one across videos
    (timestamps restart at 0) makes every video after the first fail."""
    factory = landmarker_factory or make_landmarker
    rows, errors = [], {}
    for i, r in meta.iterrows():
        lm = None
        try:
            lm = factory()
            feats = process_video(r["video_path"], lm)
            feats["participant_code"] = r["participant_code"]
            rows.append(feats)
            print(f"[3] {i+1}/{len(meta)}  {r['participant_code']}  "
                  f"face rate={feats['face_detection_rate']:.2f}")
        except Exception as e:
            errors[r["participant_code"]] = repr(e)
            print(f"[3] {i+1}/{len(meta)}  {r['participant_code']}  FAILED: {e!r}")
        finally:
            if lm is not None and hasattr(lm, "close"):
                lm.close()

    if len(rows) < max(3, int(0.5 * len(meta))):
        raise SystemExit(f"\nOnly {len(rows)}/{len(meta)} videos were processed. First errors:\n"
                         + "\n".join(f"  {k}: {v}" for k, v in list(errors.items())[:5]))
    if errors:
        print(f"[3] WARNING: {len(errors)} video(s) failed and will have empty features: {list(errors)}")

    feat = pd.DataFrame(rows)
    df = meta.drop(columns=["video_path"]).merge(feat, on="participant_code", how="left")
    df.to_csv(OUT_DIR / "face_features_raw.csv", index=False)
    bad = df.loc[~df["face_detection_quality_pass"].fillna(False).astype(bool), "participant_code"].tolist()
    if bad:
        print(f"[3] WARNING: low face-detection rate (<{MIN_DETECTION_RATE:.0%}) in: {bad}")
    print(f"[3] Saved {OUT_DIR/'face_features_raw.csv'}  shape={df.shape}")
    return df


# ----------------------------------------------------------------------------
# STEP 4 - EDA
# ----------------------------------------------------------------------------
KEY = ["face_smile_mean", "face_frown_mean", "face_brow_lowerer_mean", "face_brow_inner_raiser_mean",
       "face_jaw_open_mean", "face_eye_squint_mean", "face_disgust_proxy_mean",
       "face_expressiveness_proxy", "face_motion_energy", "face_ear_mean", "face_mar_mean",
       "face_pitch_std_deg", "face_yaw_std_deg"]


def feature_columns(df):
    skip = {"face_detected_frames", "face_detection_rate", "face_detection_quality_pass"}
    cols = [c for c in df.columns if c.startswith("face_") and c not in skip
            and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def eda(df):
    cols = feature_columns(df)
    num = df[cols].astype(float)
    kf = [c for c in KEY if c in df.columns]

    print("\n[4] ===== DATASET OVERVIEW =====")
    print("Recordings:", len(df))
    for c in ["year_level", "class_activity", "valence_group", "arousal_group", "affect_category"]:
        print(f"\n{c}:\n{df[c].value_counts().to_string()}")
    print(f"\nFace detection rate: mean={df['face_detection_rate'].mean():.2f}, "
          f"min={df['face_detection_rate'].min():.2f}; quality pass = {df['face_detection_quality_pass'].mean():.0%}")

    num.describe().T.assign(skew=num.skew()).to_csv(OUT_DIR / "descriptive_stats.csv")
    miss = num.isna().sum(); miss[miss > 0].to_csv(OUT_DIR / "missing_values.csv", header=["n_missing"])

    # label distribution
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(data=df, x="valence_group", order=["Negative", "Neutral", "Positive"], ax=ax[0])
    sns.countplot(data=df, x="arousal_group", order=["Low", "Moderate", "High"], ax=ax[1])
    sns.heatmap(pd.crosstab(df["valence_group"], df["arousal_group"]), annot=True, fmt="d", cmap="Blues", ax=ax[2])
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_label_distribution.png", dpi=150); plt.close()

    # QC: detection rate / brightness / sharpness
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, c in zip(ax, ["face_detection_rate", "video_brightness_mean", "video_sharpness_mean"]):
        sns.histplot(df[c].dropna(), kde=True, ax=a); a.set_title(c)
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_video_quality.png", dpi=150); plt.close()

    # distributions
    fig, axes = plt.subplots(3, 5, figsize=(20, 10)); axes = axes.ravel()
    for a, c in zip(axes, kf):
        sns.histplot(df[c].dropna(), kde=True, ax=a); a.set_title(c.replace("face_", ""), fontsize=9)
    for a in axes[len(kf):]:
        a.axis("off")
    plt.tight_layout(); plt.savefig(FIG_DIR / "03_feature_distributions.png", dpi=150); plt.close()

    # boxplots by emotion group
    for grp, order in [("valence_group", ["Negative", "Neutral", "Positive"]),
                       ("arousal_group", ["Low", "Moderate", "High"])]:
        fig, axes = plt.subplots(3, 5, figsize=(20, 10)); axes = axes.ravel()
        for a, c in zip(axes, kf):
            sns.boxplot(data=df, x=grp, y=c, order=order, ax=a)
            sns.stripplot(data=df, x=grp, y=c, order=order, color="k", size=3, ax=a)
            a.set_title(c.replace("face_", ""), fontsize=9); a.set_xlabel("")
        for a in axes[len(kf):]:
            a.axis("off")
        plt.tight_layout(); plt.savefig(FIG_DIR / f"04_boxplots_by_{grp}.png", dpi=150); plt.close()

    # Spearman correlations with valence / arousal
    rows = []
    for c in cols:
        for t in ["valence_score", "arousal_score"]:
            ok = df[[c, t]].dropna()
            if len(ok) > 5 and ok[c].nunique() > 1:
                rho, p = stats.spearmanr(ok[c], ok[t])
                rows.append({"feature": c, "target": t, "rho": rho, "p_value": p, "n": len(ok)})
    corr = pd.DataFrame(rows, columns=["feature", "target", "rho", "p_value", "n"]).sort_values("p_value")
    corr.to_csv(OUT_DIR / "feature_target_correlations.csv", index=False)
    print("\n[4] Top 10 Spearman correlations with valence/arousal:")
    print(corr.head(10).to_string(index=False))

    # Kruskal-Wallis
    rows = []
    for grp in ["valence_group", "arousal_group", "year_level"]:
        for c in kf:
            groups = [g[c].dropna().values for _, g in df.groupby(grp) if g[c].notna().sum() >= 2]
            if len(groups) >= 2:
                try:
                    h, p = stats.kruskal(*groups); rows.append({"grouping": grp, "feature": c, "H": h, "p_value": p})
                except ValueError:
                    pass
    pd.DataFrame(rows).sort_values("p_value").to_csv(OUT_DIR / "group_tests_kruskal.csv", index=False)

    # correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[kf].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
    plt.title("Spearman correlation - key facial features")
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_correlation_heatmap.png", dpi=150); plt.close()

    # PCA
    X = num.loc[:, num.notna().mean() > 0.5]
    X = X.fillna(X.median()); X = X.loc[:, X.std() > 0]
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit(Z); pc = pca.transform(Z)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, hue in zip(ax, ["valence_group", "arousal_group"]):
        sns.scatterplot(x=pc[:, 0], y=pc[:, 1], hue=df[hue], style=df["year_level"], s=90, ax=a)
        a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        a.set_title(f"PCA coloured by {hue}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "06_pca.png", dpi=150); plt.close()

    # outliers
    z = (df[kf] - df[kf].mean()) / df[kf].std()
    out = [(df.loc[i, "participant_code"], c, round(z.loc[i, c], 2)) for c in kf for i in z.index if abs(z.loc[i, c]) > 3]
    pd.DataFrame(out, columns=["participant", "feature", "z"]).to_csv(OUT_DIR / "outliers.csv", index=False)
    print(f"\n[4] Outlier flags (|z|>3): {len(out)} -> {OUT_DIR/'outliers.csv'}")
    return cols


# ----------------------------------------------------------------------------
# STEP 5 - MODEL-READY
# ----------------------------------------------------------------------------
def make_model_ready(df, cols):
    """Facial features only: no spoken word / text / audio columns."""
    X = df[cols].astype(float)
    X = X.drop(columns=[c for c in X.columns if c.startswith("face__neutral")], errors="ignore")
    X = X.loc[:, X.notna().mean() > 0.5]
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]                       # drops always-zero blendshapes (e.g. cheekPuff)
    Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=df.index)
    ids = df[["participant_code", "year_level", "valence_score", "arousal_score",
              "valence_group", "arousal_group", "affect_category", "face_detection_quality_pass"]]
    pd.concat([ids, X], axis=1).to_csv(OUT_DIR / "face_features_clean_unscaled.csv", index=False)
    pd.concat([ids, Z], axis=1).to_csv(OUT_DIR / "face_features_model_ready_scaled.csv", index=False)
    print(f"[5] Model-ready set: {Z.shape[1]} features x {len(Z)} samples saved to {OUT_DIR}/")


if __name__ == "__main__":
    unzip_videos()
    meta = load_metadata()
    df = run_extraction(meta)
    cols = eda(df)
    make_model_ready(df, cols)
    print("\nDone. Check the 'output_face' folder.")