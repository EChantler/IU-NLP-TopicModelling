

import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import joblib
import os
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Load transformer model from MLflow Model Registry
client = MlflowClient()
try:
    # Try to get latest version of transformer_model_prod
    versions = client.get_latest_versions("transformer_model_prod")
    if versions:
        model_version = versions[0].version
        model_uri = f"models:/transformer_model_prod/{model_version}"
    else:
        raise RuntimeError("No versions found for 'transformer_model_prod'")
    
    # Load model using MLflow's pytorch loader
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    
    # For tokenizer, we need to download artifacts manually
    version_info = client.get_model_version("transformer_model_prod", model_version)
    if version_info.run_id:
        local_model_path = mlflow.artifacts.download_artifacts(artifact_path="transformer_model", run_id=version_info.run_id)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        # Fallback: use the base model tokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        print("Warning: Using fallback tokenizer as model version has no run_id")
        
except Exception as e:
    print(f"Failed to load transformer from registry: {e}")
    # Ultimate fallback: load from local path
    print("Falling back to local transformer model path")
    model_path = './results/final_transformer_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

# --- Domain predictor ---
try:
    # Try to load domain predictor from MLflow Model Registry
    domain_versions = client.get_latest_versions("domain_model_prod")
    if domain_versions:
        domain_model_uri = f"models:/domain_model_prod/{domain_versions[0].version}"
        domain_pipe = mlflow.sklearn.load_model(domain_model_uri)
        print(f"Loaded domain predictor from MLflow registry: version {domain_versions[0].version}")
    else:
        raise RuntimeError("No versions found for 'domain_model_prod'")
except Exception as e:
    print(f"Failed to load domain predictor from registry: {e}")
    # Fallback: load from local path
    print("Falling back to local domain predictor path")
    domain_pipe_path = './results/domain_pipe.joblib'
    if not os.path.exists(domain_pipe_path):
        raise FileNotFoundError(f"Domain predictor not found at {domain_pipe_path}. Please run train_domain_predictor.py first.")
    domain_pipe = joblib.load(domain_pipe_path)

# Load label mapping from train.csv
train_df = pd.read_csv('data/processed/train.csv')
labels = sorted(train_df['target'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict_topic(text):
    pred_domain = domain_pipe.predict([text])[0]
    input_text = f"{pred_domain} [SEP] {text}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
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
