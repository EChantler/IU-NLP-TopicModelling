import re
import spacy
import pandas as pd
import os
import mlflow
from sklearn.model_selection import train_test_split

# Configuration flags
USE_SAMPLE_DATA = False  # Set to True to use only a fraction of data for testing
SAMPLE_FRACTION = 0.75    # Use 10% of data when testing

nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

URL_RE    = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE  = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
HTML_RE   = re.compile(r'<[^>]+>')
HASHTAG   = re.compile(r'#\w+')
MENTION   = re.compile(r'@\w+')
REPEAT    = re.compile(r'^([a-z])\1+$')

def clean_text_for_tfidf(text: str) -> str:
    """Aggressive cleaning for TF-IDF models"""
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

def clean_text_for_transformer(text: str) -> str:
    """Light cleaning for transformer models - preserve more structure"""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags but keep basic punctuation and structure
    text = HTML_RE.sub(' ', text)
    # Replace URLs and emails with tokens to preserve meaning
    text = URL_RE.sub(' [URL] ', text)
    text = EMAIL_RE.sub(' [EMAIL] ', text)
    # Keep hashtags and mentions but clean them
    text = HASHTAG.sub(lambda m: m.group(0).replace('#', 'hashtag_'), text)
    text = MENTION.sub(lambda m: m.group(0).replace('@', 'mention_'), text)
    
    # Clean up excessive whitespace and newlines
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse multiple spaces
    
    return text

def clean_text_for_transformer_lemmatized(text: str) -> str:
    """Moderate cleaning for transformer models with lemmatization and stop word removal"""
    if not isinstance(text, str):
        return ""
    
    # Apply light cleaning first
    text = clean_text_for_transformer(text)
    
    # Apply spaCy processing for lemmatization and stop word removal
    doc = nlp(text)
    
    # Keep lemmatized tokens, remove stop words but preserve structure
    processed_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha and len(token.lemma_) >= 2:
            processed_tokens.append(token.lemma_.lower())
        elif token.is_punct:
            # Keep some punctuation for structure
            if token.text in ['.', '!', '?', ',']:
                processed_tokens.append(token.text)
    
    return ' '.join(processed_tokens)

def filter_tokens(tokens: list[str]) -> list[str]:
    return [
        t for t in tokens
        if len(t) >= 3                # drop very short
        and not REPEAT.match(t)       # drop “tttt”
        and not t.isdigit()           # drop pure numbers
    ]

def tokenize_and_lemmatize_for_tfidf(text: str) -> list[str]:
    """Aggressive preprocessing for TF-IDF: lemmatization, stop word removal, etc."""
    if not isinstance(text, str):
        return []
    doc = nlp(clean_text_for_tfidf(text))
    
    return [
        token.lemma_
        for token in doc
        if (
            not token.is_stop
            and not token.is_punct
            and token.is_alpha
        )
    ]

def preprocess_corpus_for_tfidf(corpus: list[str]) -> list[list[str]]:
    """Process corpus for TF-IDF with aggressive cleaning"""
    processed = []
    total = len(corpus)
    for i, doc in enumerate(corpus):
        if i % 1000 == 0:
            print(f"Processing document for TF-IDF {i+1}/{total} ({(i+1)/total:.1%})")
        processed.append(filter_tokens(tokenize_and_lemmatize_for_tfidf(doc)))
    return processed

def preprocess_corpus_for_transformer(corpus: list[str]) -> list[str]:
    """Process corpus for transformers with light cleaning"""
    processed = []
    total = len(corpus)
    for i, doc in enumerate(corpus):
        if i % 1000 == 0:
            print(f"Processing document for transformer {i+1}/{total} ({(i+1)/total:.1%})")
        processed.append(clean_text_for_transformer(doc))
    return processed

def preprocess_corpus_for_transformer_lemmatized(corpus: list[str]) -> list[str]:
    """Process corpus for transformers with lemmatization and stop word removal"""
    processed = []
    total = len(corpus)
    for i, doc in enumerate(corpus):
        if i % 1000 == 0:
            print(f"Processing document for transformer (lemmatized) {i+1}/{total} ({(i+1)/total:.1%})")
        processed.append(clean_text_for_transformer_lemmatized(doc))
    return processed


# MLflow run start
mlflow.start_run()

# Log preprocessing parameters
mlflow.log_param("spacy_model", "en_core_web_sm")
mlflow.log_param("min_token_length", 3)
mlflow.log_param("remove_stop_words", True)
mlflow.log_param("lemmatization", True)
mlflow.log_param("three_cleaning_versions", True)
mlflow.log_param("use_sample_data", USE_SAMPLE_DATA)
mlflow.log_param("sample_fraction", SAMPLE_FRACTION if USE_SAMPLE_DATA else None)
mlflow.log_param("train_test_split", True)
mlflow.log_param("train_size", 0.7)
mlflow.log_param("val_size", 0.15)
mlflow.log_param("test_size", 0.15)
mlflow.log_param("stratify_by_domain", True)
mlflow.log_param("random_state", 42)
mlflow.log_param("description", f"{SAMPLE_FRACTION*100:.1f}% sample data" if USE_SAMPLE_DATA else "Full dataset")
df = pd.read_csv('./data/raw/combined_mapped.csv')

# Apply sampling if testing flag is enabled
if USE_SAMPLE_DATA:
    print(f"TESTING MODE: Using {SAMPLE_FRACTION:.1%} of the data for faster processing")
    df = df.sample(frac=SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
    print(f"Sampled dataset: {len(df)} documents")

print("Creating three versions of cleaned data...")
print(f"Dataset size: {len(df)} documents")

# Log original dataset metrics
mlflow.log_metric("original_documents_count", len(df))
mlflow.log_metric("original_domains_count", df['domain'].nunique())
mlflow.log_metric("original_targets_count", df['target'].nunique())

# Create TF-IDF version with aggressive cleaning
print("\n=== Processing for TF-IDF (aggressive cleaning) ===")
df_tfidf = df.copy()
df_tfidf['tokens'] = preprocess_corpus_for_tfidf(df_tfidf['text'].tolist())
# create a new column of space-joined text
df_tfidf["text_cleaned"] = df_tfidf["tokens"].apply(lambda toks: " ".join(toks))

# Filter out empty documents for TF-IDF
df_tfidf = df_tfidf[df_tfidf['text_cleaned'].apply(lambda x: len(x.strip()) > 0)]
print(f"TF-IDF version: {len(df_tfidf)} documents after cleaning")

# Log TF-IDF metrics
mlflow.log_metric("tfidf_documents_count", len(df_tfidf))
mlflow.log_metric("tfidf_documents_filtered", len(df) - len(df_tfidf))

# Split TF-IDF data into train/val/test (70/15/15)
train_tfidf, temp_tfidf = train_test_split(
    df_tfidf[["text_cleaned", "target", "domain"]], 
    test_size=0.3, 
    random_state=42, 
    stratify=df_tfidf['domain']
)

# Split temp into val (15%) and test (15%)
val_tfidf, test_tfidf = train_test_split(
    temp_tfidf, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_tfidf['domain']
)

# Log TF-IDF split metrics
mlflow.log_metric("tfidf_train_count", len(train_tfidf))
mlflow.log_metric("tfidf_val_count", len(val_tfidf))
mlflow.log_metric("tfidf_test_count", len(test_tfidf))

# Save TF-IDF version
os.makedirs('./data/processed', exist_ok=True)
train_tfidf.rename(columns={"text_cleaned": "text"}).to_csv(
    "./data/processed/train_tfidf.csv", index=False
)
val_tfidf.rename(columns={"text_cleaned": "text"}).to_csv(
    "./data/processed/val_tfidf.csv", index=False
)
test_tfidf.rename(columns={"text_cleaned": "text"}).to_csv(
    "./data/processed/test_tfidf.csv", index=False
)
print("Saved: ./data/processed/train_tfidf.csv")
print("Saved: ./data/processed/val_tfidf.csv")
print("Saved: ./data/processed/test_tfidf.csv")

# Create Transformer version with light cleaning  
print("\n=== Processing for Transformers (light cleaning) ===")
df_transformer = df.copy()
df_transformer['text_cleaned'] = preprocess_corpus_for_transformer(df_transformer['text'].tolist())

# Filter out empty documents for transformers
df_transformer = df_transformer[df_transformer['text_cleaned'].apply(lambda x: len(x.strip()) > 0)]
print(f"Transformer version: {len(df_transformer)} documents after cleaning")

# Log Transformer metrics
mlflow.log_metric("transformer_documents_count", len(df_transformer))
mlflow.log_metric("transformer_documents_filtered", len(df) - len(df_transformer))

# Split Transformer data into train/val/test (70/15/15)
train_transformer, temp_transformer = train_test_split(
    df_transformer[["text_cleaned", "target", "domain"]], 
    test_size=0.3, 
    random_state=42, 
    stratify=df_transformer['domain']
)

# Split temp into val (15%) and test (15%)
val_transformer, test_transformer = train_test_split(
    temp_transformer, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_transformer['domain']
)

# Log Transformer split metrics
mlflow.log_metric("transformer_train_count", len(train_transformer))
mlflow.log_metric("transformer_val_count", len(val_transformer))
mlflow.log_metric("transformer_test_count", len(test_transformer))

# Save Transformer version
train_transformer.rename(columns={"text_cleaned": "text"}).to_csv(
    "./data/processed/train_transformer.csv", index=False
)
val_transformer.rename(columns={"text_cleaned": "text"}).to_csv(
    "./data/processed/val_transformer.csv", index=False
)
test_transformer.rename(columns={"text_cleaned": "text"}).to_csv(
    "./data/processed/test_transformer.csv", index=False
)
print("Saved: ./data/processed/train_transformer.csv")
print("Saved: ./data/processed/val_transformer.csv")
print("Saved: ./data/processed/test_transformer.csv")

# Create Transformer Lemmatized version with moderate cleaning
print("\n=== Processing for Transformers (lemmatized + stop word removal) ===")
df_transformer_lem = df.copy()
df_transformer_lem['text_cleaned'] = preprocess_corpus_for_transformer_lemmatized(df_transformer_lem['text'].tolist())

# Filter out empty documents for transformer lemmatized
df_transformer_lem = df_transformer_lem[df_transformer_lem['text_cleaned'].apply(lambda x: len(x.strip()) > 0)]
print(f"Transformer lemmatized version: {len(df_transformer_lem)} documents after cleaning")

# Log Transformer lemmatized metrics
mlflow.log_metric("transformer_lem_documents_count", len(df_transformer_lem))
mlflow.log_metric("transformer_lem_documents_filtered", len(df) - len(df_transformer_lem))

# Split Transformer lemmatized data into train/val/test (70/15/15)
train_transformer_lem, temp_transformer_lem = train_test_split(
    df_transformer_lem[["text_cleaned", "target", "domain"]], 
    test_size=0.3, 
    random_state=42, 
    stratify=df_transformer_lem['domain']
)

# Split temp into val (15%) and test (15%)
val_transformer_lem, test_transformer_lem = train_test_split(
    temp_transformer_lem, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_transformer_lem['domain']
)

# Log Transformer lemmatized split metrics
mlflow.log_metric("transformer_lem_train_count", len(train_transformer_lem))
mlflow.log_metric("transformer_lem_val_count", len(val_transformer_lem))
mlflow.log_metric("transformer_lem_test_count", len(test_transformer_lem))

# Save Transformer lemmatized version
train_transformer_lem.rename(columns={"text_cleaned": "text"}).to_csv(
    "./data/processed/train_transformer_lem.csv", index=False
)
val_transformer_lem.rename(columns={"text_cleaned": "text"}).to_csv(
    "./data/processed/val_transformer_lem.csv", index=False
)
test_transformer_lem.rename(columns={"text_cleaned": "text"}).to_csv(
    "./data/processed/test_transformer_lem.csv", index=False
)
print("Saved: ./data/processed/train_transformer_lem.csv")
print("Saved: ./data/processed/val_transformer_lem.csv")
print("Saved: ./data/processed/test_transformer_lem.csv")

print(f"\nSummary:")
print(f"Original: {len(df)} documents")
print(f"TF-IDF version: {len(df_tfidf)} documents")
print(f"  - Train: {len(train_tfidf)} documents")
print(f"  - Val: {len(val_tfidf)} documents")
print(f"  - Test: {len(test_tfidf)} documents")
print(f"Transformer version: {len(df_transformer)} documents")
print(f"  - Train: {len(train_transformer)} documents")
print(f"  - Val: {len(val_transformer)} documents")
print(f"  - Test: {len(test_transformer)} documents")
print(f"Transformer lemmatized version: {len(df_transformer_lem)} documents")
print(f"  - Train: {len(train_transformer_lem)} documents")
print(f"  - Val: {len(val_transformer_lem)} documents")
print(f"  - Test: {len(test_transformer_lem)} documents")

# Log final summary metrics
mlflow.log_metric("final_tfidf_retention_rate", len(df_tfidf) / len(df))
mlflow.log_metric("final_transformer_retention_rate", len(df_transformer) / len(df))
mlflow.log_metric("final_transformer_lem_retention_rate", len(df_transformer_lem) / len(df))

# Log split ratios as parameters (strings) and metrics (numeric)
mlflow.log_param("train_val_test_ratio", f"{len(train_tfidf)}:{len(val_tfidf)}:{len(test_tfidf)}")
mlflow.log_metric("train_percentage", len(train_tfidf) / len(df_tfidf))
mlflow.log_metric("val_percentage", len(val_tfidf) / len(df_tfidf))
mlflow.log_metric("test_percentage", len(test_tfidf) / len(df_tfidf))

# Log artifacts to MLflow
processed_dir = "./data/processed"
mlflow.log_artifacts(processed_dir, artifact_path="cleaned_data")

# Show some examples
print(f"\nExample comparisons (from training data):")
sample_idx = 0
if len(df) > sample_idx:
    print(f"\nOriginal text (first 200 chars):")
    print(f"{df.iloc[sample_idx]['text'][:200]}...")
    
    if len(train_tfidf) > sample_idx:
        print(f"\nTF-IDF cleaned (train):")
        # Use text_cleaned since we haven't renamed it yet in the dataframe
        tfidf_text = train_tfidf.iloc[sample_idx]['text_cleaned'] if len(train_tfidf) > sample_idx else "N/A"
        print(f"{tfidf_text[:200]}...")
    
    if len(train_transformer) > sample_idx:
        print(f"\nTransformer cleaned (train):")
        # Use text_cleaned since we haven't renamed it yet in the dataframe
        transformer_text = train_transformer.iloc[sample_idx]['text_cleaned'] if len(train_transformer) > sample_idx else "N/A"
        print(f"{transformer_text[:200]}...")
    
    if len(train_transformer_lem) > sample_idx:
        print(f"\nTransformer lemmatized cleaned (train):")
        # Use text_cleaned since we haven't renamed it yet in the dataframe
        transformer_lem_text = train_transformer_lem.iloc[sample_idx]['text_cleaned'] if len(train_transformer_lem) > sample_idx else "N/A"
        print(f"{transformer_lem_text[:200]}...")

# End MLflow run
mlflow.end_run()
