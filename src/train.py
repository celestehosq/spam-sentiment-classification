from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from scipy.sparse import csr_matrix, hstack


VADER = SentimentIntensityAnalyzer()


def load_data(path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Expected columns: text, label in processed CSV")
    x = df["text"].astype(str)
    y = df["label"].astype(str).str.lower().str.strip()
    return x, y


def vader_features(texts: pd.Series) -> csr_matrix:
    rows = []
    for t in texts.fillna("").astype(str):
        s = VADER.polarity_scores(t)
        compound_01 = (s["compound"] + 1.0) / 2.0  # shift from [-1,1] -> [0,1]
        rows.append([s["neg"], s["neu"], s["pos"], compound_01])
    return csr_matrix(rows)


def evaluate(name: str, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    print(f"\n===== {name} =====")
    print(classification_report(y_test, preds, digits=4))
    cm = confusion_matrix(y_test, preds, labels=["ham", "spam"])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    print("Confusion matrix [ham, spam]:")
    print(cm)
    return {"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/enron.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--out", type=str, default="reports/results.csv")
    args = ap.parse_args()

    x, y = load_data(Path(args.data))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=args.max_features,
    )

    X_train_tfidf = tfidf.fit_transform(x_train)
    X_test_tfidf = tfidf.transform(x_test)

    X_train_sent = vader_features(x_train)
    X_test_sent = vader_features(x_test)

    X_train_combo = hstack([X_train_tfidf, X_train_sent])
    X_test_combo = hstack([X_test_tfidf, X_test_sent])

    models = [
        ("NB_tfidf", MultinomialNB(), X_train_tfidf, X_test_tfidf),
        ("RF_tfidf", RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1),
         X_train_tfidf, X_test_tfidf),
        ("MLP_tfidf", MLPClassifier(hidden_layer_sizes=(128,), max_iter=8, random_state=args.seed),
         X_train_tfidf, X_test_tfidf),

        ("NB_tfidf_sent", MultinomialNB(), X_train_combo, X_test_combo),
        ("RF_tfidf_sent", RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1),
         X_train_combo, X_test_combo),
        ("MLP_tfidf_sent", MLPClassifier(hidden_layer_sizes=(128,), max_iter=8, random_state=args.seed),
         X_train_combo, X_test_combo),
    ]

    results = []
    for name, clf, Xtr, Xte in models:
        clf.fit(Xtr, y_train)
        results.append(evaluate(name, clf, Xte, y_test))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

