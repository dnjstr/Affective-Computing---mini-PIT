import glob
import os

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# SETTINGS
# ============================================================

VIDEO_DIR = "Videos"
LABELS_CSV = "G4_-_RESEARCH_MINI-PROJECT_v2_-_G4_-_2nd_Year.csv"

OUT_CSV = "facial_feature_dataset.csv"        # 1 row per participant
FRAME_CSV = "facial_frame_features.csv"       # 1 row per video frame

EAR_BLINK_THRESHOLD = 0.20   # eye aspect ratio below this = eye closed


# ============================================================
# CHECK FILES AND FOLDERS
# ============================================================

if not os.path.exists(VIDEO_DIR):
    raise FileNotFoundError(
        f'Video folder "{VIDEO_DIR}" was not found. '
        f"Make sure it is inside the same folder as this Python file."
    )

if not os.path.exists(LABELS_CSV):
    raise FileNotFoundError(
        f'CSV file "{LABELS_CSV}" was not found. '
        f"Make sure it is inside the same folder as this Python file."
    )


# ============================================================
# MEDIAPIPE FACE MESH LANDMARK INDICES (468 + 10 iris points)
# ============================================================

# Eye corners (outer, inner) -> used for inter-ocular distance (IOD)
R_OUTER, R_INNER = 33, 133
L_OUTER, L_INNER = 263, 362

# Eye aspect ratio points: p1(outer), p2, p3 (upper), p4(inner), p5, p6 (lower)
R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]

# Mouth
LIP_UP_IN, LIP_LOW_IN = 13, 14
MOUTH_L_IN, MOUTH_R_IN = 78, 308
MOUTH_L_OUT, MOUTH_R_OUT = 61, 291

# Eyebrows: (brow top point, eye upper-lid point)
R_BROW, R_LID = 105, 159
L_BROW, L_LID = 334, 386
R_BROW_INNER, L_BROW_INNER = 107, 336

# Iris centres (needs refine_landmarks=True)
R_IRIS, L_IRIS = 468, 473

# Head pose points + generic 3D face model (mm)
POSE_IDX = [1, 152, 33, 263, 61, 291]
POSE_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # eye outer (image left)
    (225.0, 170.0, -135.0),   # eye outer (image right)
    (-150.0, -150.0, -125.0), # mouth corner (image left)
    (150.0, -150.0, -125.0),  # mouth corner (image right)
], dtype=np.float64)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def ear(pts):
    """Eye Aspect Ratio (Soukupova & Cech, 2016)."""
    p1, p2, p3, p4, p5, p6 = pts
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-9)


def head_pose(lm_px, w, h):
    """Return pitch, yaw, roll in degrees (0,0,0 = facing camera)."""
    img_pts = np.array([lm_px[i] for i in POSE_IDX], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        POSE_3D, img_pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return np.nan, np.nan, np.nan
    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    pitch, yaw, roll = angles
    # remove the +/-180 flip that solvePnP sometimes returns
    pitch = ((pitch + 90) % 180) - 90
    roll = ((roll + 90) % 180) - 90
    return pitch, yaw, roll


# ============================================================
# PER-FRAME FEATURES
# ============================================================

def frame_features(lm, w, h):
    """
    lm: (478, 3) array of normalised landmarks.
    All distances are divided by the inter-ocular distance (IOD)
    so they do not depend on how close the person is to the camera.
    """
    px = lm[:, :2] * np.array([w, h])
    iod = dist(px[R_OUTER], px[L_OUTER])

    f = {}
    f["face_size_iod_px"] = iod

    # ---- Eyes ------------------------------------------------
    ear_r = ear([px[i] for i in R_EYE])
    ear_l = ear([px[i] for i in L_EYE])
    f["ear_right"], f["ear_left"] = ear_r, ear_l
    f["ear_mean"] = (ear_r + ear_l) / 2

    # ---- Mouth -----------------------------------------------
    mouth_open = dist(px[LIP_UP_IN], px[LIP_LOW_IN])
    mouth_w_in = dist(px[MOUTH_L_IN], px[MOUTH_R_IN])
    f["mar"] = mouth_open / (mouth_w_in + 1e-9)               # mouth aspect ratio
    f["mouth_open_iod"] = mouth_open / iod
    f["mouth_width_iod"] = dist(px[MOUTH_L_OUT], px[MOUTH_R_OUT]) / iod  # smile = wider

    # smile lift: how far the mouth corners sit above the lip centre
    lip_mid_y = (px[LIP_UP_IN][1] + px[LIP_LOW_IN][1]) / 2
    corner_y = (px[MOUTH_L_OUT][1] + px[MOUTH_R_OUT][1]) / 2
    f["smile_lift_iod"] = (lip_mid_y - corner_y) / iod

    # ---- Eyebrows --------------------------------------------
    f["brow_raise_right"] = dist(px[R_BROW], px[R_LID]) / iod
    f["brow_raise_left"] = dist(px[L_BROW], px[L_LID]) / iod
    f["brow_raise_mean"] = (f["brow_raise_right"] + f["brow_raise_left"]) / 2
    f["brow_inner_dist_iod"] = dist(px[R_BROW_INNER], px[L_BROW_INNER]) / iod  # small = furrowed

    # ---- Gaze (iris position inside the eye, 0..1) -----------
    def gaze_h(iris, outer, inner):
        return (px[iris][0] - px[outer][0]) / (px[inner][0] - px[outer][0] + 1e-9)

    if lm.shape[0] >= 478:
        f["gaze_h_right"] = gaze_h(R_IRIS, R_OUTER, R_INNER)
        f["gaze_h_left"] = gaze_h(L_IRIS, L_OUTER, L_INNER)
        f["gaze_h_mean"] = (f["gaze_h_right"] + f["gaze_h_left"]) / 2
        # vertical: iris y between upper lid and lower lid (0 = up, 1 = down)
        gv_r = (px[R_IRIS][1] - px[160][1]) / (px[144][1] - px[160][1] + 1e-9)
        gv_l = (px[L_IRIS][1] - px[385][1]) / (px[380][1] - px[385][1] + 1e-9)
        f["gaze_v_mean"] = (gv_r + gv_l) / 2
    else:
        f["gaze_h_right"] = f["gaze_h_left"] = f["gaze_h_mean"] = np.nan
        f["gaze_v_mean"] = np.nan

    # ---- Head pose -------------------------------------------
    f["head_pitch"], f["head_yaw"], f["head_roll"] = head_pose(px, w, h)

    return f


def normalised_shape(lm, w, h):
    """Landmarks centred on the eye midpoint and scaled by IOD
    (used to measure facial movement independent of position/zoom)."""
    px = lm[:, :2] * np.array([w, h])
    centre = (px[R_OUTER] + px[L_OUTER]) / 2
    iod = dist(px[R_OUTER], px[L_OUTER])
    return (px - centre) / iod


# ============================================================
# EXTRACT FEATURES FROM ONE VIDEO
# ============================================================

def extract(path, face_mesh):
    code = os.path.splitext(os.path.basename(path))[0]
    print(f"Processing: {os.path.basename(path)}")

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows, shapes = [], []
    n_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1

        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not res.multi_face_landmarks:
            continue

        lm = np.array(
            [[p.x, p.y, p.z] for p in res.multi_face_landmarks[0].landmark]
        )
        feats = frame_features(lm, w, h)
        feats["participant_code"] = code
        feats["frame"] = n_frames - 1
        feats["time_sec"] = (n_frames - 1) / fps
        rows.append(feats)
        shapes.append(normalised_shape(lm, w, h))

    cap.release()

    frames_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Basic video information
    # --------------------------------------------------------
    r = {
        "participant_code": code,
        "video_filename": os.path.basename(path),
        "video_fps": fps,
        "video_width": w,
        "video_height": h,
        "video_frames": n_frames,
        "video_duration_sec": n_frames / fps,
        "face_frames": len(frames_df),
        "face_detect_ratio": len(frames_df) / max(n_frames, 1),
    }

    if frames_df.empty:
        return r, frames_df

    # --------------------------------------------------------
    # Statistics per feature (same style as audio: mean/std/max/min)
    # --------------------------------------------------------
    feature_cols = [
        c for c in frames_df.columns
        if c not in ("participant_code", "frame", "time_sec")
    ]

    for c in feature_cols:
        a = frames_df[c].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            a = np.array([np.nan])
        r[f"face_{c}_mean"] = np.nanmean(a)
        r[f"face_{c}_std"] = np.nanstd(a)
        r[f"face_{c}_max"] = np.nanmax(a)
        r[f"face_{c}_min"] = np.nanmin(a)

    # --------------------------------------------------------
    # Blinks / eye closure
    # --------------------------------------------------------
    closed = frames_df["ear_mean"].to_numpy() < EAR_BLINK_THRESHOLD
    r["face_eye_closed_ratio"] = float(closed.mean())
    r["face_blink_count"] = int(np.sum(np.diff(closed.astype(int)) == 1))

    # --------------------------------------------------------
    # Facial / head movement
    # --------------------------------------------------------
    if len(shapes) >= 2:
        S = np.stack(shapes)                                  # (T, 478, 2)
        disp = np.linalg.norm(np.diff(S, axis=0), axis=2)     # (T-1, 478)
        r["face_motion_mean"] = float(disp.mean())
        r["face_motion_max"] = float(disp.mean(axis=1).max())

        # mouth-region movement only (expression, not head motion)
        mouth_idx = [61, 291, 13, 14, 78, 308, 0, 17]
        r["face_mouth_motion_mean"] = float(disp[:, mouth_idx].mean())
        brow_idx = [105, 107, 66, 334, 336, 296]
        r["face_brow_motion_mean"] = float(disp[:, brow_idx].mean())
    else:
        r["face_motion_mean"] = r["face_motion_max"] = np.nan
        r["face_mouth_motion_mean"] = r["face_brow_motion_mean"] = np.nan

    # Head movement range
    for ang in ("pitch", "yaw", "roll"):
        a = frames_df[f"head_{ang}"].to_numpy()
        r[f"face_head_{ang}_range"] = float(np.nanmax(a) - np.nanmin(a))

    # Change over the clip (last third minus first third): did the smile grow/fade?
    third = max(len(frames_df) // 3, 1)
    for c in ("mouth_width_iod", "smile_lift_iod", "mar", "brow_raise_mean"):
        v = frames_df[c].to_numpy()
        r[f"face_{c}_delta"] = float(np.nanmean(v[-third:]) - np.nanmean(v[:third]))

    return r, frames_df


# ============================================================
# BUILD LABEL TEMPLATE (identical to the audio script)
# ============================================================

def build_template(labels_csv):
    raw = pd.read_csv(labels_csv)
    raw.columns = raw.columns.astype(str).str.replace("**", "", regex=False).str.strip()

    o = pd.DataFrame()
    o["participant_code"] = raw["Participant Code"].astype(str).str.strip()
    o["year_level"] = raw["Year Level"]
    o["class_activity"] = raw["Class Activity"]
    o["subject"] = raw["Subject"]
    o["session"] = raw["Session"].map({"Morning": "AM", "Afternoon": "PM"})
    o["session_name"] = raw["Session"]
    o["spoken_word"] = raw["Spoken Word"]
    o["self_reported_feeling"] = raw["Self-Reported Feeling"]

    for col, name in [("Valence (1-5)", "valence"), ("Arousal (1-5)", "arousal")]:
        values = raw[col].astype(str).str.replace("**", "", regex=False).str.strip()
        parts = values.str.split(" — ", n=1, expand=True)
        o[f"{name}_score"] = parts[0].astype(int)
        o[f"{name}_label"] = parts[1].str.strip()

    o["affect_category"] = ""
    o["word_english"] = ""
    o["date_collected"] = raw["Date Collected"]
    return o


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("==========================================")
    print(" FACIAL FEATURE EXTRACTION")
    print("==========================================")

    files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"\nFound {len(files)} video files.")

    if not files:
        raise FileNotFoundError(f'No .mp4 files found inside "{VIDEO_DIR}".')

    feature_rows, frame_dfs = [], []

    for file in files:
        # new FaceMesh per video so tracking does not leak between people
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as fm:
            try:
                row, fdf = extract(file, fm)
                feature_rows.append(row)
                frame_dfs.append(fdf)
            except Exception as e:
                print(f"\nERROR processing {file}:\n{e}")

    if not feature_rows:
        raise RuntimeError("No facial features were successfully extracted.")

    feats = pd.DataFrame(feature_rows)
    print(f"\nSuccessfully extracted features from {len(feats)} videos.")

    # frame-level file (useful for time-series plots)
    pd.concat(frame_dfs, ignore_index=True).round(5).to_csv(FRAME_CSV, index=False)
    print(f"Saved {FRAME_CSV}")

    template = build_template(LABELS_CSV)
    out = template.merge(feats, on="participant_code", how="left", validate="1:1")

    missing = out["video_filename"].isna().sum()
    if missing:
        print(f"\nWARNING: {missing} participants have no matching video.")
        for c in out.loc[out["video_filename"].isna(), "participant_code"]:
            print(f"  - {c}")
    else:
        print("\nAll participants have matching video files.")

    low = out[out["face_detect_ratio"] < 0.9]
    if len(low):
        print("\nWARNING: face detected in <90% of frames for:")
        print(low[["participant_code", "face_detect_ratio"]].to_string(index=False))

    out.round(5).to_csv(OUT_CSV, index=False)

    print("\n==========================================")
    print(f"Saved {OUT_CSV}")
    print(f"Rows:    {out.shape[0]}")
    print(f"Columns: {out.shape[1]}")
    print("==========================================")