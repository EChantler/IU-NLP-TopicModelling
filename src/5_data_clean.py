import re
import spacy
import pandas as pd
import os

nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

URL_RE    = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE  = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
HTML_RE   = re.compile(r'<[^>]+>')
HASHTAG   = re.compile(r'#\w+')
MENTION   = re.compile(r'@\w+')
REPEAT    = re.compile(r'^([a-z])\1+$')

def clean_text(text: str) -> str:
    text = text.lower()
    text = HTML_RE.sub(' ', text)
    text = URL_RE.sub(' ', text)
    text = EMAIL_RE.sub(' ', text)
    text = HASHTAG.sub(' ', text)
    text = MENTION.sub(' ', text)
    # keep letters+digits; drop everything else
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_tokens(tokens: list[str]) -> list[str]:
    return [
        t for t in tokens
        if len(t) >= 3                # drop very short
        and not REPEAT.match(t)       # drop “tttt”
        and not t.isdigit()           # drop pure numbers
    ]

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
    processed = []
    total = len(corpus)
    for i, doc in enumerate(corpus):
        if i % 1000 == 0:
            print(f"Processing document {i+1}/{total} ({(i+1)/total:.1%})")
        processed.append(filter_tokens(tokenize_and_lemmatize(doc)))
    return processed


df = pd.read_csv('./data/raw/combined_mapped.csv')

df['tokens'] = preprocess_corpus(df['text'].tolist())
# create a new column of space-joined text
df["text"] = df["tokens"].apply(lambda toks: " ".join(toks))

# save only the text + labels
if 'tokens' in df.columns:
    df = df.drop(columns=['tokens'])
df[["text","target"]].to_csv("./data/processed/combined_cleaned.csv", index=False)

print(f"Processed {len(df)} documents.")
df = df[df['text'].apply(lambda x: len(x) > 0)]
print(f"Filtered to {len(df)} documents after cleaning.")
# Ensure the output directory exists
os.makedirs('./data/processed', exist_ok=True)
df.to_csv('./data/processed/combined_cleaned.csv', index=False)
