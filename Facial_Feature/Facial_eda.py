import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# SETTINGS
# ============================================================
FACE_CSV = "facial_feature_dataset.csv"
FRAME_CSV = "facial_frame_features.csv"
RESULT_SUBDIR = "facial_feature_dataset_result"   # fallback location of the CSVs
AUDIO_CSV = os.path.join("..", "Audio_Feature", "audio_feature_dataset_result",
                         "audio_feature_dataset.csv") 
FIG_DIR = "EDA_figures"
OUT_DIR = "EDA_results"                           
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def load_valid(name, need_rows=True):
    """Return the first copy of `name` that really contains feature values."""
    for p in (name, os.path.join(RESULT_SUBDIR, name)):
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        has_values = d.filter(like="face_").notna().any().any()
        if has_values:
            print(f"[load] using {p}")
            return d
        print(f"[load] SKIPPING {p}: no feature values in it (empty/NaN)")
    raise SystemExit(
        f"\nNo usable '{name}' found. Re-run Extract_Facial_Feature.py and read the "
        "'WARNING: ... no matching video' lines, or copy the file from "
        f"{RESULT_SUBDIR}/ into this folder.")


face = load_valid(FACE_CSV).copy()
frames = load_valid(FRAME_CSV)

KEY = {
    "face_ear_mean_mean": "Eye openness (EAR)",
    "face_mar_mean": "Mouth openness (MAR)",
    "face_mouth_width_iod_mean": "Mouth width (smile)",
    "face_smile_lift_iod_mean": "Smile lift",
    "face_brow_raise_mean_mean": "Brow raise",
    "face_brow_inner_dist_iod_mean": "Inner-brow distance (furrow)",
    "face_gaze_h_mean_mean": "Gaze horizontal",
    "face_head_pitch_mean": "Head pitch",
    "face_head_yaw_mean": "Head yaw",
    "face_head_roll_mean": "Head roll",
    "face_blink_count": "Blink count",
    "face_motion_mean": "Facial motion",
    "face_mouth_motion_mean": "Mouth motion",
    "face_brow_motion_mean": "Brow motion",
    "face_mouth_width_iod_delta": "Smile change over clip",
}
KEY = {k: v for k, v in KEY.items() if k in face.columns}
TARGETS = {"valence_score": "Valence", "arousal_score": "Arousal"}

report = []
def log(s=""):
    print(s); report.append(s)

# ============================================================
# 1. DATA QUALITY
# ============================================================
log("=" * 60); log("1. DATA QUALITY"); log("=" * 60)
log(f"Participants: {len(face)}")
log(f"Videos with a face in every frame: {(face.face_detect_ratio == 1).sum()}")
log(f"Min detection ratio: {face.face_detect_ratio.min():.3f}")
log(f"Total frames analysed: {int(face.face_frames.sum())}")
log(f"Clip length (s): mean {face.video_duration_sec.mean():.2f}, "
    f"min {face.video_duration_sec.min():.2f}, max {face.video_duration_sec.max():.2f}")
log(f"Missing values in feature columns: "
    f"{int(face.filter(like='face_').isna().sum().sum())}")

# ============================================================
# 2. LABEL DISTRIBUTION
# ============================================================
log("\n" + "=" * 60); log("2. LABEL DISTRIBUTION"); log("=" * 60)
log(pd.crosstab(face.class_activity, face.valence_score).to_string())
log("")
log(pd.crosstab(face.class_activity, face.arousal_score).to_string())

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
sns.countplot(x="valence_score", data=face, ax=ax[0], color="#4C72B0"); ax[0].set_title("Valence")
sns.countplot(x="arousal_score", data=face, ax=ax[1], color="#DD8452"); ax[1].set_title("Arousal")
sns.scatterplot(data=face, x="valence_score", y="arousal_score", hue="class_activity",
                s=140, alpha=.7, ax=ax[2])
ax[2].set_title("Valence vs Arousal"); ax[2].set_xlim(0.5, 5.5); ax[2].set_ylim(0.5, 5.5)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/01_label_distribution.png", dpi=150); plt.close()

# ============================================================
# 3. FEATURE DISTRIBUTIONS
# ============================================================
log("\n" + "=" * 60); log("3. DESCRIPTIVE STATISTICS (key features)"); log("=" * 60)
desc = face[list(KEY)].describe().T[["mean", "std", "min", "50%", "max"]]
desc.index = [KEY[i] for i in desc.index]
log(desc.round(3).to_string())
desc.round(4).to_csv(os.path.join(OUT_DIR, "eda_descriptive_stats.csv"))

n = len(KEY); cols = 5; rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.7 * rows))
for a, (k, name) in zip(axes.ravel(), KEY.items()):
    sns.histplot(face[k], kde=len(face) > 5, ax=a, color="#55A868"); a.set_title(name, fontsize=9); a.set_xlabel("")
for a in axes.ravel()[n:]: a.axis("off")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/02_feature_distributions.png", dpi=150); plt.close()

# ============================================================
# 4. OUTLIERS (|z| > 2.5)
# ============================================================
log("\n" + "=" * 60); log("4. OUTLIERS (|z| > 2.5)"); log("=" * 60)
z = (face[list(KEY)] - face[list(KEY)].mean()) / face[list(KEY)].std(ddof=0)
found = False
for k in KEY:
    for i in z.index[z[k].abs() > 2.5]:
        log(f"{face.participant_code[i]}: {KEY[k]} z={z.loc[i, k]:.2f}"); found = True
if not found: log("None")

# ============================================================
# 5. FEATURES vs AFFECT (Spearman)
# ============================================================
log("\n" + "=" * 60); log("5. SPEARMAN CORRELATION WITH VALENCE / AROUSAL"); log("=" * 60)
rows_ = []
for k, name in KEY.items():
    r = {"feature": name}
    for t, tn in TARGETS.items():
        rho, p = stats.spearmanr(face[k], face[t])
        r[f"{tn}_rho"], r[f"{tn}_p"] = rho, p
    rows_.append(r)
corr = pd.DataFrame(rows_).set_index("feature")
log(corr.round(3).to_string())
corr.round(4).to_csv(os.path.join(OUT_DIR, "eda_feature_affect_correlation.csv"))
log("\nNOTE: n=25 and 15 features tested -> p-values are NOT corrected for multiple "
    "comparisons. Treat as exploratory.")

fig, ax = plt.subplots(figsize=(5.5, 6))
sns.heatmap(corr[["Valence_rho", "Arousal_rho"]], annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-.7, vmax=.7, ax=ax, cbar_kws={"label": "Spearman ρ"})
ax.set_title("Facial features vs self-reported affect"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/03_feature_affect_correlation.png", dpi=150); plt.close()

# ============================================================
# 6. GROUP DIFFERENCES: valence groups and class activity
# ============================================================
log("\n" + "=" * 60); log("6. GROUP COMPARISONS"); log("=" * 60)
face["valence_group"] = pd.cut(face.valence_score, [0, 2, 3, 5],
                               labels=["Unpleasant(1-2)", "Neutral(3)", "Pleasant(4-5)"])
log(face.valence_group.value_counts().to_string())
log("\nMean by valence group:")
log(face.groupby("valence_group", observed=True)[list(KEY)[:6]].mean()
    .rename(columns=KEY).round(3).T.to_string())

log("\nQuiz (2nd yr) vs Examination (4th yr) - Mann-Whitney U")
a = face[face.class_activity == "Quiz"]; b = face[face.class_activity == "Examination"]
cmp_rows = []
for k, name in KEY.items():
    u, p = stats.mannwhitneyu(a[k], b[k], alternative="two-sided")
    cmp_rows.append((name, a[k].mean(), b[k].mean(), p))
    log(f"{name:32s} quiz={a[k].mean():8.3f}  exam={b[k].mean():8.3f}  p={p:.3f}")
log("CAUTION: class activity is fully confounded with year level and subject "
    "(all Quiz = Y2, all Exam = Y4).")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a_, k in zip(ax, ["face_mouth_width_iod_mean", "face_smile_lift_iod_mean", "face_mar_mean"]):
    sns.boxplot(data=face, x="valence_group", y=k, ax=a_, color="#CCB974", showfliers=False)
    sns.stripplot(data=face, x="valence_group", y=k, ax=a_, color="k", size=5, alpha=.7)
    a_.set_title(KEY[k]); a_.set_xlabel(""); a_.tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/04_features_by_valence_group.png", dpi=150); plt.close()

# ============================================================
# 7. REDUNDANCY BETWEEN FEATURES
# ============================================================
log("\n" + "=" * 60); log("7. FEATURE REDUNDANCY (|Spearman| > 0.7)"); log("=" * 60)
cm = face[list(KEY)].rename(columns=KEY).corr(method="spearman")
seen = False
for i, c1 in enumerate(cm.columns):
    for c2 in cm.columns[i + 1:]:
        if abs(cm.loc[c1, c2]) > 0.7:
            log(f"{c1} <-> {c2}: {cm.loc[c1, c2]:.2f}"); seen = True
if not seen: log("None above 0.7")
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(cm, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, annot=True, fmt=".1f",
            annot_kws={"size": 7})
ax.set_title("Facial feature correlation matrix (Spearman)")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/05_feature_correlation_matrix.png", dpi=150); plt.close()

# ============================================================
# 8. PCA
# ============================================================
log("\n" + "=" * 60); log("8. PCA"); log("=" * 60)
X = StandardScaler().fit_transform(face[list(KEY)].fillna(face[list(KEY)].median()))
pca = PCA(n_components=min(5, X.shape[1])).fit(X)
log("Explained variance ratio: " + ", ".join(f"PC{i+1}={v:.2f}" for i, v in enumerate(pca.explained_variance_ratio_)))
pcs = pca.transform(X)
load = pd.DataFrame(pca.components_[:2].T, index=[KEY[k] for k in KEY], columns=["PC1", "PC2"])
log("Top loadings:\n" + load.reindex(load.abs().max(axis=1).sort_values(ascending=False).index).head(6).round(2).to_string())
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=face.valence_score, cmap="RdYlGn", s=120, edgecolor="k")
for i, c in enumerate(face.participant_code): ax.annotate(c[-3:], (pcs[i, 0], pcs[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.colorbar(sc, label="Valence"); ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.set_title("PCA of facial features")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/06_pca.png", dpi=150); plt.close()

# ============================================================
# 9. TEMPORAL PATTERN (frame-level)
# ============================================================
fr = frames.merge(face[["participant_code", "valence_group"]], on="participant_code")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a_, k, t in zip(ax, ["mouth_width_iod", "mar", "brow_raise_mean"],
                    ["Mouth width (smile)", "Mouth openness (MAR)", "Brow raise"]):
    sns.lineplot(data=fr, x="time_sec", y=k, hue="valence_group", errorbar="se", ax=a_)
    a_.set_title(t); a_.set_xlim(0, 1.2)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/07_temporal_by_valence.png", dpi=150); plt.close()

# ============================================================
# 10. MULTIMODAL CHECK (optional - needs the audio dataset)
# ============================================================
if os.path.exists(AUDIO_CSV):
    log("\n" + "=" * 60); log("10. FACE vs AUDIO (same participants)"); log("=" * 60)
    au = pd.read_csv(AUDIO_CSV)
    m = face.merge(au[["participant_code", "audio_rms_mean", "audio_f0_mean", "audio_voiced_duration_sec"]],
                   on="participant_code")
    for fcol in ["face_mouth_motion_mean", "face_mar_mean", "face_motion_mean"]:
        for acol in ["audio_rms_mean", "audio_f0_mean"]:
            rho, p = stats.spearmanr(m[fcol], m[acol], nan_policy="omit")
            log(f"{KEY.get(fcol, fcol):22s} vs {acol:18s} rho={rho:6.2f} p={p:.3f}")
    m.to_csv(os.path.join(OUT_DIR, "multimodal_face_audio_dataset.csv"), index=False)

with open(os.path.join(OUT_DIR, "eda_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"\nFigures saved in ./{FIG_DIR}/ , report in ./{OUT_DIR}/")