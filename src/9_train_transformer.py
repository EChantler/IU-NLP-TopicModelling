import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset
import numpy as np

# Load data
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')

# We'll use the predicted domain as an additional feature, but transformers only take text.
# So, concatenate domain to the text for each sample.
def add_domain_to_text(df):
    return df['domain'] + ' [SEP] ' + df['text']

train_df['input_text'] = add_domain_to_text(train_df)
val_df['input_text'] = add_domain_to_text(val_df)

# Encode labels
labels = sorted(train_df['target'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
train_df['label'] = train_df['target'].map(label2id)
val_df['label'] = val_df['target'].map(label2id)

# Huggingface Dataset
train_ds = Dataset.from_pandas(train_df[['input_text', 'label']])
val_ds = Dataset.from_pandas(val_df[['input_text', 'label']])

# Model & Tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['input_text'], truncation=True, padding='max_length', max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
cols = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=cols)
val_ds.set_format(type='torch', columns=cols)

# Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

# Training args
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the model and tokenizer
model_save_path = './results/final_transformer_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved to {model_save_path}")

# Evaluate
eval_results = trainer.evaluate()
print('Validation Results:', eval_results)

# Classification report
preds = trainer.predict(val_ds)
pred_labels = np.argmax(preds.predictions, axis=1)
print(classification_report(val_ds['label'], pred_labels, target_names=labels))
