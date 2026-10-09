import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,precision_recall_fscore_support, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_utils import audio_columns, facial_columns

OUT = ROOT / "Multimodal_Feature" / "output_multimodal"
(OUT / "figures").mkdir(exist_ok=True)
SEED = 41
K_FEATURES = 10

df = pd.read_csv(OUT / "multimodal_features_unimputed.csv")
TARGETS = {
    "valence": ((df["valence_score"] >= 4).astype(int), ["Not positive", "Positive"]),
    "arousal": ((df["arousal_score"] >= 4).astype(int), ["Not high", "High"]),
}
FEATURES = {
    "Audio": audio_columns(df),
    "Facial": facial_columns(df),
    "Multimodal": audio_columns(df) + facial_columns(df),
}


def prep():
    return [SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            VarianceThreshold()]


MODELS = {
    "Majority baseline": lambda: DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": lambda: make_pipeline(
        *prep(), StandardScaler(), SelectKBest(f_classif, k=K_FEATURES),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)),
    "Random Forest": lambda: make_pipeline(
        *prep(), SelectKBest(f_classif, k=K_FEATURES),
        RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=2, class_weight="balanced", random_state=SEED)),
}

rows, matrices = [], {}
for tname, (y, labels) in TARGETS.items():
    n_splits = min(5, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fname, cols in FEATURES.items():
        for mname, make in MODELS.items():
            pred = cross_val_predict(make(), df[cols], y, cv=cv)
            p, r, f1, _ = precision_recall_fscore_support(
                y, pred, average="macro", zero_division=0)
            pc, rc, fc, _ = precision_recall_fscore_support(
                y, pred, labels=[0, 1], zero_division=0)
            rows.append({"target": tname, "features": fname, "model": mname,
                        "accuracy": accuracy_score(y, pred),
                        "balanced_accuracy": balanced_accuracy_score(y, pred),
                        "precision_macro": p, "recall_macro": r, "f1_macro": f1,
                        "positive_class": labels[1], "precision_pos": pc[1],
                        "recall_pos": rc[1], "f1_pos": fc[1],
                        "n": len(y), "folds": n_splits})
            matrices[(tname, fname, mname)] = confusion_matrix(y, pred, labels=[0, 1])

res = pd.DataFrame(rows)
res.to_csv(OUT / "classification_results.csv", index=False)

for tname, (y, labels) in TARGETS.items():
    fig, axes = plt.subplots(len(FEATURES), 2, figsize=(7, 9))
    for i, fname in enumerate(FEATURES):
        for j, mname in enumerate(["Logistic Regression", "Random Forest"]):
            cm = matrices[(tname, fname, mname)]
            ax = axes[i, j]
            ax.imshow(cm, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, cm[a, b], ha="center", va="center",
                            color="white" if cm[a, b] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], labels, fontsize=8)
            ax.set_yticks([0, 1], labels, fontsize=8)
            ax.set_title(f"{fname} | {mname}", fontsize=9)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
    fig.suptitle(f"{tname}: out-of-fold confusion matrices (n=25)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"09_confusion_matrices_{tname}.png", dpi=160)
    plt.close(fig)

pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))