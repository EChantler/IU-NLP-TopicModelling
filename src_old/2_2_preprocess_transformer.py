#%% Load the data for raw
from datetime import datetime
import pandas as pd
import re

train_df = pd.read_csv('../data/raw/train.csv')
val_df = pd.read_csv('../data/raw/val.csv')
test_df = pd.read_csv('../data/raw/test.csv')
# %% Light preprocessing for transformer models
def clean_text_transformer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)           # remove emails
    text = re.sub(r'\s+', ' ', text).strip()        # normalize whitespace
    return text

# Light preprocessing for transformer models (no lemmatization, no stopword removal)
train_df['text'] = train_df['text'].apply(clean_text_transformer)
val_df['text'] = val_df['text'].apply(clean_text_transformer)
test_df['text'] = test_df['text'].apply(clean_text_transformer)

# Save the lightly preprocessed data to csv files
train_df.to_csv('../data/processed/train.csv', index=False)
val_df.to_csv('../data/processed/val.csv', index=False)
test_df.to_csv('../data/processed/test.csv', index=False)

# %%
