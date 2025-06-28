#%% Load the data for raw
import pandas as pd
train_df = pd.read_csv('../data/raw/train.csv')
val_df = pd.read_csv('../data/raw/val.csv')
test_df = pd.read_csv('../data/raw/test.csv')

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

def preprocess_corpus(corpus: list[str]) -> list[list[str]]:
    return [tokenize_and_lemmatize(doc) for doc in corpus]


# %% Preprocess the train, validation, and test data
train_df['text'] = preprocess_corpus(train_df['text'].tolist())
val_df['text'] = preprocess_corpus(val_df['text'].tolist())
test_df['text'] = preprocess_corpus(test_df['text'].tolist())
# %% Save the preprocessed data to csv files
train_df.to_csv('../data/processed/train.csv', index=False)
val_df.to_csv('../data/processed/val.csv', index=False)
test_df.to_csv('../data/processed/test.csv', index=False)

# %% Load the preprocessed data
train_df = pd.read_csv('../data/processed/train.csv')
val_df = pd.read_csv('../data/processed/val.csv')
test_df = pd.read_csv('../data/processed/test.csv')

# %% Print the shape of the preprocessed data
print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

# %%
