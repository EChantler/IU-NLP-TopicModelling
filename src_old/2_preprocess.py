#%% Load the data for raw
from datetime import datetime
import pandas as pd

train_df = pd.read_csv('./data/raw/train.csv')
val_df = pd.read_csv('./data/raw/val.csv')
test_df = pd.read_csv('./data/raw/test.csv')

#%% Clean and preprocess the text data
import re
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_and_lemmatize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    doc = nlp(clean_text(text))
    
    return [
        token.lemma_
        for token in doc
        if (
            not token.is_stop
            and not token.is_punct
            and token.is_alpha
        )
    ]

RE_REPEAT = re.compile(r'^([a-z])\1+$')

def filter_single_chars_and_repeats(tokens):
    """Filter out single character tokens and repeated characters."""
    if not isinstance(tokens, list):
        return []
    # Filter out single character tokens (length 1) and repeated characters (e.g., "aaa", "bb")
    tokens = [t for t in tokens if len(t) > 1]
    # Filter out tokens that are just repeated characters
    return [t for t in tokens if not RE_REPEAT.match(t)]

def preprocess_corpus(corpus: list[str]) -> list[list[str]]:
    return [filter_single_chars_and_repeats(tokenize_and_lemmatize(doc)) for doc in corpus]


# %% Preprocess the train, validation, and test data
train_df['text'] = preprocess_corpus(train_df['text'].tolist())  # Limiting to 1000 samples for faster processing
val_df['text'] = preprocess_corpus(val_df['text'].tolist())
test_df['text'] = preprocess_corpus(test_df['text'].tolist())
# %% Save the preprocessed data to csv files
train_df.to_csv('./data/processed/train.csv', index=False)
val_df.to_csv('./data/processed/val.csv', index=False)
test_df.to_csv('./data/processed/test.csv', index=False)

# %% Load the preprocessed data
train_df = pd.read_csv('./data/processed/train.csv')
val_df = pd.read_csv('./data/processed/val.csv')
test_df = pd.read_csv('./data/processed/test.csv')

# %% Print the shape of the preprocessed data
print(f"{datetime.now()}\n")
print(f"Preprocessed Data Shapes:")
print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

# %%
