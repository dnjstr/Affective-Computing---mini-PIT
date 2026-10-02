import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)
warnings.filterwarnings("ignore")

# ------------------------------- CONFIG ------------------------------------
FACE_CSV  = "Facial_Feature/facial_feature_dataset_result/facial_feature_dataset.csv"
AUDIO_CSV = "Audio_Feature/audio_feature_dataset_result/audio_feature_dataset.csv"

OUT_DIRS = {
    "face":       "Facial_Feature/Model_results",
    "audio":      "Audio_Feature/Model_results",
    "face+audio": "Multimodal_results",
}
# ---------------------------------------------------------------------------

MODELS = {
    "Majority baseline": lambda: DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": lambda: make_pipeline(
        SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
    "Random Forest": lambda: make_pipeline(
        SimpleImputer(), RandomForestClassifier(300, random_state=0)),
    "SVM (RBF)": lambda: make_pipeline(
        SimpleImputer(), StandardScaler(), SVC(C=1.0, kernel="rbf")),
    "kNN (k=3)": lambda: make_pipeline(
        SimpleImputer(), StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
}

# 3-level targets (valence has only 1 sample at score <= 2, so 'neg' is tiny)
TARGETS = {
    "valence": lambda s: s.map(lambda v: "negative" if v <= 2 else ("neutral" if v == 3 else "positive")),
    "arousal": lambda s: s.map(lambda v: "low" if v <= 2 else ("moderate" if v == 3 else "high")),
}


def load():
    face = pd.read_csv(FACE_CSV)
    audio = pd.read_csv(AUDIO_CSV)
    face_cols = [c for c in face.columns if c.startswith("face_")
                 and pd.api.types.is_numeric_dtype(face[c])]
    audio_cols = [c for c in audio.columns if c.startswith("audio_")
                  and pd.api.types.is_numeric_dtype(audio[c])]
    df = face[["participant_code", "valence_score", "arousal_score"] + face_cols].merge(
        audio[["participant_code"] + audio_cols], on="participant_code", validate="1:1")
    return df, {"face": face_cols, "audio": audio_cols, "face+audio": face_cols + audio_cols}


def main():
    df, feature_sets = load()
    labels = {t: f(df[f"{t}_score"]) for t, f in TARGETS.items()}

    for fs_name, cols in feature_sets.items():
        out = OUT_DIRS[fs_name]
        os.makedirs(out, exist_ok=True)
        rows, preds = [], pd.DataFrame({"participant_code": df.participant_code})
        report = [f"Feature set: {fs_name}  ({len(cols)} features, n={len(df)}, LOOCV)\n"]

        for t, y in labels.items():
            report.append(f"\n--- Target: {t} | class counts: {y.value_counts().to_dict()}")
            for m_name, make in MODELS.items():
                p = cross_val_predict(make(), df[cols], y, cv=LeaveOneOut())
                acc = accuracy_score(y, p)
                f1 = f1_score(y, p, average="macro")
                rows.append({"target": t, "model": m_name,
                             "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
                preds[f"{t}__{m_name}"] = p
                report.append(f"{m_name:20s} acc={acc:.3f}  macroF1={f1:.3f}")

                # confusion matrix image per model/target
                fig, ax = plt.subplots(figsize=(4.5, 4))
                classes = sorted(y.unique())
                ConfusionMatrixDisplay(confusion_matrix(y, p, labels=classes),
                                       display_labels=classes).plot(ax=ax, colorbar=False)
                ax.set_title(f"{fs_name} | {t} | {m_name}\nacc={acc:.2f}", fontsize=9)
                fig.tight_layout()
                safe = m_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
                fig.savefig(os.path.join(out, f"cm_{t}_{safe}.png"), dpi=150)
                plt.close(fig)

        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(out, "accuracy_results.csv"), index=False)
        preds.to_csv(os.path.join(out, "predictions.csv"), index=False)

        # accuracy bar chart (dashed line = majority-class baseline)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, t in zip(axes, TARGETS):
            r = res[res.target == t]
            ax.bar(r.model, r.accuracy)
            ax.axhline(r[r.model == "Majority baseline"].accuracy.iloc[0], ls="--", c="k")
            ax.set_title(f"{fs_name}: {t}")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=30)
        axes[0].set_ylabel("LOOCV accuracy")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "accuracy_comparison.png"), dpi=150)
        plt.close(fig)

        with open(os.path.join(out, "model_report.txt"), "w") as f:
            f.write("\n".join(report))
        print("\n".join(report), f"\n -> saved to {out}/\n")


if __name__ == "__main__":
    main()