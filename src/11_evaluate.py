
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score

import joblib


# Load transformer model and tokenizer
model_path = './results/final_transformer_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# --- Domain predictor ---
domain_pipe_path = './results/domain_pipe.joblib'
if not os.path.exists(domain_pipe_path):
    raise FileNotFoundError(f"Domain predictor not found at {domain_pipe_path}. Please run train_domain_predictor.py first.")
domain_pipe = joblib.load(domain_pipe_path)


# Predict domain for each row and add to input_text
def add_predicted_domain_to_text(df):
    pred_domains = domain_pipe.predict(df['text'])
    return pred_domains + ' [SEP] ' + df['text']


def evaluate_on_file(test_path, label2id, id2label, labels):
    df = pd.read_csv(test_path)
    print(f"Evaluating on {test_path} with {len(df)} samples...")
    # Predict domain from text
    df['input_text'] = add_predicted_domain_to_text(df)
    print(f"Predicted domains for {len(df)} samples.")
    df['label'] = df['target'].map(label2id)
    test_ds = Dataset.from_pandas(df[['input_text', 'label']])
    def tokenize(batch):
        return tokenizer(batch['input_text'], truncation=True, padding='max_length', max_length=256)
    test_ds = test_ds.map(tokenize, batched=True)
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # Run inference
    all_logits = []
    all_labels = []
    with torch.no_grad():
        print(f"Running inference for {test_path} ({len(test_ds)} samples)...")
        for batch in test_ds.iter(batch_size=16):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels_ = batch['label']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels_.cpu().numpy())
    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)
    y_pred = np.argmax(logits, axis=1)
    acc = accuracy_score(y_true, y_pred)
    print(f"\nResults for {test_path}:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0
    ))

# Get label mappings from train set
train_df = pd.read_csv('data/processed/train.csv')
labels = sorted(train_df['target'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# Evaluate on each test file
test_files = [
    'data/processed/test_news.csv',
    'data/processed/test_social.csv',
    'data/processed/test_academic.csv',
    'data/processed/test.csv',
]
for test_path in test_files:
    evaluate_on_file(test_path, label2id, id2label, labels)
