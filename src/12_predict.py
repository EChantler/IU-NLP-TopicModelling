

import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import joblib
import os
from text_cleaning import clean_text

# Load domain predictor
domain_pipe_path = './results/domain_pipe.joblib'
if not os.path.exists(domain_pipe_path):
    raise FileNotFoundError(f"Domain predictor not found at {domain_pipe_path}. Please run train_domain_predictor.py first.")
domain_pipe = joblib.load(domain_pipe_path)

# Load model and tokenizer
model_path = './results/final_transformer_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load label mapping from train.csv
train_df = pd.read_csv('data/processed/train.csv')
labels = sorted(train_df['target'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


# Load domain predictor
domain_pipe_path = './results/domain_pipe.joblib'
if not os.path.exists(domain_pipe_path):
    raise FileNotFoundError(f"Domain predictor not found at {domain_pipe_path}. Please run train_domain_predictor.py first.")
domain_pipe = joblib.load(domain_pipe_path)

def predict_topic(text):
    # Clean text for domain prediction (must match training)
    cleaned_text = clean_text(text)
    pred_domain = domain_pipe.predict([cleaned_text])[0]
    input_text = f"{pred_domain} [SEP] {text}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        return id2label[pred_id], pred_domain


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'your text here'")
        sys.exit(1)
    text = sys.argv[1]
    topic, pred_domain = predict_topic(text)
    print(f"Predicted domain: {pred_domain}")
    print(f"Predicted topic: {topic}")
