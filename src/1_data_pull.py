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


def load_arxiv_classification():
    from datasets import load_dataset

    ds = load_dataset("gfissore/arxiv-abstracts-2021")
    # Use 'abstract' as text and the first part of the first category as target
    data = ds["train"]["abstract"]
    target = [cats[0].split()[0].split(".")[0] if cats else "unknown" for cats in ds["train"]["categories"]]
    return data, target


def load_reddit_topics():
    from datasets import load_dataset

    ds = load_dataset("jamescalam/reddit-topics")
    data = ds["train"]["selftext"]
    target = ds["train"]["sub"]
    return data, target


def execute(dataset="arxiv_classification"):
    print(f"Loading dataset: {dataset}")
    loaders = {
        "20newsgroup": (load_20newsgroups, "news.csv"),
        "reuters": (load_reuters, "news.csv"),
        "imdb": (load_imdb, "reviews.csv"),
        "ag_news": (load_ag_news, "news.csv"),
        "arxiv_classification": (load_arxiv_classification, "academic.csv"),
        "reddit_topics": (load_reddit_topics, "social.csv"),
    }
    if dataset not in loaders:
        raise ValueError(f"Dataset {dataset} not supported.")
    loader, out_file = loaders[dataset]
    data, target = loader()

    out_dir = f"./data/raw/"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"text": data, "target": target}).to_csv(
        f"{out_dir}/{out_file}", index=False
    )
    print(f"Data saved to {out_dir}/{out_file}")


if __name__ == "__main__":
    execute()