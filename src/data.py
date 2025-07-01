import os
from sklearn.model_selection import train_test_split
import pandas as pd


def load_20newsgroups():
    from sklearn.datasets import fetch_20newsgroups

    newsgroups = fetch_20newsgroups(
        data_home="../data",
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    data = newsgroups.data
    target = newsgroups.target
    target_names = newsgroups.target_names
    target = [target_names[t] for t in target]
    return data, target


def load_reuters():
    import nltk

    nltk.download("reuters")
    from nltk.corpus import reuters

    docs = reuters.fileids()
    data = [reuters.raw(doc_id) for doc_id in docs]
    target = [
        reuters.categories(doc_id)[0] if reuters.categories(doc_id) else "unknown"
        for doc_id in docs
    ]
    return data, target


def load_imdb():
    from datasets import load_dataset

    ds = load_dataset("imdb")
    label_names = ds["train"].features["label"].names
    data = ds["train"]["text"] + ds["test"]["text"]
    target = [label_names[label] for label in ds["train"]["label"]] + [
        label_names[label] for label in ds["test"]["label"]
    ]
    return data, target


def load_ag_news():
    from datasets import load_dataset

    ds = load_dataset("ag_news")
    label_names = ds["train"].features["label"].names
    data = ds["train"]["text"] + ds["test"]["text"]
    target = [label_names[label] for label in ds["train"]["label"]] + [
        label_names[label] for label in ds["test"]["label"]
    ]
    return data, target


def execute(dataset="ag_news"):
    print(f"Loading dataset: {dataset}")
    loaders = {
        "20newsgroup": load_20newsgroups,
        "reuters": load_reuters,
        "imdb": load_imdb,
        "ag_news": load_ag_news,
    }
    if dataset not in loaders:
        raise ValueError(f"Dataset {dataset} not supported.")
    data, target = loaders[dataset]()

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    out_dir = f"./data/raw/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"text": X_train, "target": y_train}).to_csv(
        f"{out_dir}/train.csv", index=False
    )
    pd.DataFrame({"text": X_val, "target": y_val}).to_csv(
        f"{out_dir}/val.csv", index=False
    )
    pd.DataFrame({"text": X_test, "target": y_test}).to_csv(
        f"{out_dir}/test.csv", index=False
    )
    print(f"Data saved to {out_dir}/")


if __name__ == "__main__":
    execute()