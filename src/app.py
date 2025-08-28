import argparse, os, json, joblib, requests, io, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import get_paths

RAW_PATH = "data/raw/SMSSpamCollection.tsv"
URL = "https://raw.githubusercontent.com/justmarkham/DAT5/master/data/SMSSpamCollection.txt"

def ensure_dirs():
    for d in ["data/raw", "data/interim", "data/processed", "models", "reports/figures"]:
        os.makedirs(d, exist_ok=True)

def download_dataset_if_needed():
    ensure_dirs()
    if os.path.exists(RAW_PATH):
        return pd.read_csv(RAW_PATH, sep="\t", names=["label","text"], header=None)
    r = requests.get(URL)
    df = pd.read_csv(io.StringIO(r.text), sep="\t", names=["label","text"], header=None)
    df.to_csv(RAW_PATH, sep="\t", index=False, header=False)
    return df

def clean_df(df):
    df = df.dropna().copy()
    df["label"] = df["label"].str.lower().str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["label"].isin(["ham","spam"])]

def build_pipeline():
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
    grid = GridSearchCV(pipe, {"clf__C": [0.5, 1.0, 2.0]}, cv=5, scoring="f1_macro", n_jobs=-1)
    return grid

def plot_confusion(cm, labels, path):
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.savefig(path)
    plt.close(fig)

def train_and_eval():
    model_path, vec_path, metrics_path = get_paths()
    df = clean_df(download_dataset_if_needed())
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], stratify=df["label"], test_size=0.2)
    grid = build_pipeline()
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=["ham","spam"])
    plot_confusion(cm, ["ham","spam"], "reports/figures/confusion_matrix.png")

    joblib.dump(best, model_path)
    joblib.dump(best.named_steps["tfidf"], vec_path)

    with open(metrics_path, "w") as f:
        json.dump({"best_params": grid.best_params_, "report": report}, f, indent=2)

    print("Entrenamiento completado")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if args.train or args.eval:
        train_and_eval()

