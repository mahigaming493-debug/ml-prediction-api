# %% [markdown]
# # Breast Cancer Prediction: End-to-End ML Lifecycle
# VS Code me `# %%` wale cells "Run Cell" se alag-alag chal sakte hain (Jupyter extension chahiye),
# ya pura file: `python train.py`

# %% 1. PROBLEM DEFINITION
# Goal: tumor ke measurements se predict karna ki wo Malignant (0) hai ya Benign (1).
# Type: Binary classification. Primary metric: Recall of malignant class
# (cancer miss karna sabse costly galti hai), saath me F1 / ROC-AUC.

# %% 2. DATA COLLECTION
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "model"
REPORTS.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SEED = 42

raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()          # 'target': 0 = malignant, 1 = benign
df.to_csv(ROOT / "data.csv", index=False)
print("Shape:", df.shape)

# %% 3. EDA
print(df.describe().T.head(10))
print("\nClass balance:\n", df["target"].value_counts(normalize=True).rename({0: "malignant", 1: "benign"}))

fig, ax = plt.subplots(figsize=(4, 3))
df["target"].map({0: "malignant", 1: "benign"}).value_counts().plot.bar(ax=ax, color=["#c0392b", "#27ae60"])
ax.set_title("Class distribution"); plt.tight_layout(); plt.savefig(REPORTS / "01_class_balance.png", dpi=120); plt.close()

plt.figure(figsize=(11, 9))
sns.heatmap(df.drop(columns="target").corr(), cmap="coolwarm", center=0)
plt.title("Feature correlation"); plt.tight_layout(); plt.savefig(REPORTS / "02_correlation.png", dpi=120); plt.close()

# %% 4. DATA CLEANING
print("\nMissing values:", int(df.isna().sum().sum()))
print("Duplicates:", int(df.duplicated().sum()))
df = df.drop_duplicates().dropna()

# Outliers: IQR check (sirf report; tree/scaled models robust hain, isliye rows drop nahi kar rahe)
X_all = df.drop(columns="target")
q1, q3 = X_all.quantile(0.25), X_all.quantile(0.75)
iqr = q3 - q1
outlier_share = ((X_all < q1 - 1.5 * iqr) | (X_all > q3 + 1.5 * iqr)).any(axis=1).mean()
print(f"Rows with at least one IQR outlier: {outlier_share:.1%} (kept)")

# %% 5. FEATURE ENGINEERING / SELECTION
# 30 features form me daalna practical nahi, isliye top 6 chunte hain
# (mutual information, correlated duplicates avoid karne ke liye 'worst' + 'mean' mix check karke).
y = df["target"]
mi = pd.Series(mutual_info_classif(X_all, y, random_state=SEED), index=X_all.columns).sort_values(ascending=False)
print("\nTop MI features:\n", mi.head(10))

selected, corr = [], X_all.corr().abs()
for feat in mi.index:
    if all(corr.loc[feat, s] < 0.90 for s in selected):   # highly correlated feature skip
        selected.append(feat)
    if len(selected) == 6:
        break
print("Selected features:", selected)
X = X_all[selected]

# %% 6. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# %% 7. MODEL TRAINING (baseline comparison, 5-fold CV)
candidates = {
    "LogisticRegression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]),
    "SVC": Pipeline([("sc", StandardScaler()), ("m", SVC(probability=True, random_state=SEED))]),
    "RandomForest": Pipeline([("m", RandomForestClassifier(n_estimators=200, random_state=SEED))]),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cv_scores = {}
for name, pipe in candidates.items():
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    cv_scores[name] = float(s.mean())
    print(f"{name:20s} CV F1 = {s.mean():.4f} (+/- {s.std():.4f})")

# %% 8. HYPERPARAMETER TUNING (GridSearchCV)
best_name = max(cv_scores, key=cv_scores.get)
print("\nBest baseline:", best_name)
grids = {
    "LogisticRegression": {"m__C": [0.01, 0.1, 1, 10, 100]},
    "SVC": {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.01, 0.1]},
    "RandomForest": {"m__n_estimators": [100, 300], "m__max_depth": [None, 5, 10]},
}
search = GridSearchCV(candidates[best_name], grids[best_name], cv=cv, scoring="f1", n_jobs=-1)
search.fit(X_train, y_train)
model = search.best_estimator_
print("Best params:", search.best_params_, "| CV F1:", round(search.best_score_, 4))

# %% 9. EVALUATION (test set, sirf ek baar)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
metrics = {
    "model": best_name,
    "best_params": search.best_params_,
    "cv_f1": round(float(search.best_score_), 4),
    "test_f1": round(float(f1_score(y_test, pred)), 4),
    "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    "test_recall_malignant": round(float(recall_score(y_test, pred, pos_label=0)), 4),
    "baseline_cv_f1": cv_scores,
}
print("\n", classification_report(y_test, pred, target_names=["malignant", "benign"]))
print(json.dumps(metrics, indent=2))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["malignant", "benign"], cmap="Blues")
plt.title("Confusion matrix (test)"); plt.tight_layout(); plt.savefig(REPORTS / "03_confusion_matrix.png", dpi=120); plt.close()
RocCurveDisplay.from_predictions(y_test, proba)
plt.title("ROC curve (test)"); plt.tight_layout(); plt.savefig(REPORTS / "04_roc_curve.png", dpi=120); plt.close()

# %% 10. SAVE MODEL + METADATA
joblib.dump(model, MODEL_DIR / "model.pkl")
meta = {
    "features": selected,
    "target_mapping": {"0": "malignant", "1": "benign"},
    "feature_ranges": {f: [float(X[f].min()), float(X[f].max())] for f in selected},
    "metrics": metrics,
}
(MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
print("\nSaved model/model.pkl and model/metadata.json")

# %% 11. SMOKE TEST: saved model reload karke prediction
loaded = joblib.load(MODEL_DIR / "model.pkl")
sample = X_test.iloc[[0]]
print("Sample:", sample.to_dict("records")[0])
print("Prediction:", raw.target_names[int(loaded.predict(sample)[0])], "| actual:", raw.target_names[int(y_test.iloc[0])])

# %% 12. MONITORING (production me kya karte hain, notes)
# - Input drift: incoming feature distributions ko training ranges (metadata.json) se compare karo.
# - Prediction drift: benign/malignant ratio me achanak badlav dekho.
# - Performance: ground-truth labels aane par F1/recall dobara nikalo, girne par retrain.
# - API me out-of-range inputs par warning bheji gayi hai (api/index.py).
