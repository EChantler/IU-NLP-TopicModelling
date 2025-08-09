import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# MLflow imports
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Load transformer model from MLflow Model Registry
# Use the latest version (fallback if no Production stage)
client = MlflowClient()
try:
    # Try to get Production version first
    versions = client.get_latest_versions("transformer_model_prod")
    if versions:
        model_version = versions[0].version
        model_uri = f"models:/transformer_model/{model_version}"
    else:
        raise RuntimeError("No versions found for 'transformer_model'")
    
    # Load model using MLflow's pytorch loader
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    
    # For tokenizer, we need to download artifacts manually
    version_info = client.get_model_version("transformer_model", model_version)
    if version_info.run_id:
        local_model_path = mlflow.artifacts.download_artifacts(artifact_path="transformer_model", run_id=version_info.run_id)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        # Fallback: use the base model tokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        print("Warning: Using fallback tokenizer as model version has no run_id")
        
except Exception as e:
    print(f"Failed to load from registry: {e}")
    # Ultimate fallback: load from local path
    print("Falling back to local model path")
    model_path = './results/final_transformer_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

# Detect device and move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Helper to add domain to text
def add_domain_to_text(df):
    return df['domain'] + ' [SEP] ' + df['text']

# Helper to get misclassified samples
def get_misclassified(df, label2id, id2label, labels, set_name):
    df['input_text'] = add_domain_to_text(df)
    df['label'] = df['target'].map(label2id)
    ds = Dataset.from_pandas(df[['input_text', 'label']])
    def tokenize(batch):
        return tokenizer(batch['input_text'], truncation=True, padding='max_length', max_length=256)
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    all_logits = []
    all_labels = []
    total = len(ds)
    print(f"Running inference for {set_name} ({total} samples)...")
    for i, batch in enumerate(ds.iter(batch_size=16)):
        if i % 20 == 0:
            print(f"  Processed {i*16} / {total} samples ({(i*16/total)*100:.1f}%)")
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_ = batch['label']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels_.cpu().numpy())
    print(f"Finished inference for {set_name}.")
    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)
    y_pred = np.argmax(logits, axis=1)
    # Find misclassified indices
    mis_idx = np.where(y_true != y_pred)[0]
    mis_df = df.iloc[mis_idx].copy()
    mis_df['true_label'] = [id2label[i] for i in y_true[mis_idx]]
    mis_df['pred_label'] = [id2label[i] for i in y_pred[mis_idx]]
    print(f"\nMisclassified samples from {set_name} (showing up to 20):")
    print(mis_df[['text', 'domain', 'true_label', 'pred_label']].head(20))
    

    # assuming y_true, y_pred for your validation set and `labels` is the list of all class names
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    plt.figure(figsize=(6,6))
    disp.plot(cmap='Blues', xticks_rotation=45, values_format='d')
    plt.show()
    
    return mis_df

# Get label mappings from train set
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')
labels = sorted(train_df['target'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# Show misclassified samples for train and val sets
get_misclassified(train_df, label2id, id2label, labels, 'train set')
get_misclassified(val_df, label2id, id2label, labels, 'validation set')
