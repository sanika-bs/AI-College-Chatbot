"""Train TF-IDF + Logistic Regression intent classifier and write real evaluation reports."""
import json, os
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from preprocessing import preprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "dataset", "college_faq_dataset.csv")
MODELS, REPORTS = os.path.join(ROOT, "models"), os.path.join(ROOT, "reports")
SEED = 42

def make_vectorizer():
    # word 1-2 grams capture phrases ("admission procedure"); char n-grams add robustness to misspellings
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
    ])

def main():
    os.makedirs(MODELS, exist_ok=True); os.makedirs(REPORTS, exist_ok=True)
    df = pd.read_csv(DATA)
    df["cleaned_question"] = df["question"].apply(preprocess)          # same function used at prediction time
    df.to_csv(os.path.join(REPORTS, "dataset_with_cleaned_text.csv"), index=False)
    X_tr, X_te, y_tr, y_te = train_test_split(df["cleaned_question"], df["intent"], test_size=0.2,
                                              stratify=df["intent"], random_state=SEED)
    print(f"Samples: {len(df)} | train: {len(X_tr)} | test: {len(X_te)} | intents: {df['intent'].nunique()}")

    # ---- model comparison via 5-fold CV on the training split only
    candidates = {"LogisticRegression": LogisticRegression(C=10, max_iter=3000),
                  "LinearSVC": LinearSVC(C=1), "MultinomialNB": MultinomialNB(alpha=0.1)}
    vec = make_vectorizer(); Xtr = vec.fit_transform(X_tr)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED); comparison = {}
    for name, clf in candidates.items():
        s = cross_val_score(clf, Xtr, y_tr, cv=cv, scoring="accuracy")
        comparison[name] = {"cv_accuracy_mean": round(float(s.mean()), 4), "cv_accuracy_std": round(float(s.std()), 4)}
        print(f"  CV {name:20s} {s.mean():.3f} +/- {s.std():.3f}")

    # ---- final evaluation of Logistic Regression on the held-out test set
    clf = LogisticRegression(C=10, max_iter=3000).fit(Xtr, y_tr)
    pred = clf.predict(vec.transform(X_te))
    labels = sorted(df["intent"].unique())
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
    metrics = {"model": "TF-IDF (word 1-2 + char 2-4 grams) + LogisticRegression", "train_size": len(X_tr),
               "test_size": len(X_te), "accuracy": round(accuracy_score(y_te, pred), 4),
               "precision_weighted": round(p, 4), "recall_weighted": round(r, 4), "f1_weighted": round(f1, 4),
               "model_comparison_cv": comparison}
    rep = classification_report(y_te, pred, labels=labels, zero_division=0)
    open(os.path.join(REPORTS, "classification_report.txt"), "w").write(rep)
    json.dump(metrics, open(os.path.join(REPORTS, "metrics.json"), "w"), indent=2)
    cm = confusion_matrix(y_te, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(REPORTS, "confusion_matrix.csv"))
    fig, ax = plt.subplots(figsize=(9, 7.5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right"); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=8)
    ax.set_xlabel("Predicted intent"); ax.set_ylabel("True intent"); ax.set_title("Confusion matrix (held-out test set)")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "confusion_matrix.png"), dpi=130); plt.close()
    print("\nHeld-out test results\n" + rep); print(json.dumps({k: v for k, v in metrics.items() if k != "model_comparison_cv"}, indent=2))

    # ---- deployment model: refit on ALL data (metrics above come from the train/test split model)
    vec_full = make_vectorizer(); Xall = vec_full.fit_transform(df["cleaned_question"])
    clf_full = LogisticRegression(C=10, max_iter=3000).fit(Xall, df["intent"])
    joblib.dump(vec_full, os.path.join(MODELS, "tfidf_vectorizer.joblib"))
    joblib.dump(clf_full, os.path.join(MODELS, "intent_model.joblib"))
    print("Saved models/tfidf_vectorizer.joblib and models/intent_model.joblib")

if __name__ == "__main__": main()
