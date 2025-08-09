import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import os

mlflow.start_run()

# Load data from MLflow artifacts
client = MlflowClient()
try:
    # Get all experiments and search for data_splits artifacts
    experiments = client.search_experiments()
    data_run = None
    
    for experiment in experiments:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=10)
        for run in runs:
            try:
                artifacts = client.list_artifacts(run.info.run_id, path="data_splits")
                if artifacts:  # Found a run with data_splits
                    data_run = run
                    print(f"Found data_splits in experiment {experiment.name} (ID: {experiment.experiment_id})")
                    break
            except Exception:
                continue
        if data_run:
            break
    
    if data_run:
        print(f"Loading data from MLflow run: {data_run.info.run_id}")
        # Download data artifacts
        data_path = mlflow.artifacts.download_artifacts(artifact_path="data_splits", run_id=data_run.info.run_id)
        train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_path, 'val.csv'))
        print(f"Loaded data from MLflow artifacts")
    else:
        raise FileNotFoundError("No runs with data_splits artifacts found in any experiment")
        
except Exception as e:
    print(f"Failed to load data from MLflow: {e}")
    print("Falling back to local data files")
    # Fallback to local files
    train_df = pd.read_csv('../data/processed/train.csv')
    val_df = pd.read_csv('../data/processed/val.csv')
  
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
# Log training parameters
mlflow.log_param('model_name', model_name)
mlflow.log_param('learning_rate', training_args.learning_rate)
mlflow.log_param('train_batch_size', training_args.per_device_train_batch_size)
mlflow.log_param('eval_batch_size', training_args.per_device_eval_batch_size)
mlflow.log_param('num_train_epochs', training_args.num_train_epochs)
mlflow.log_param('weight_decay', training_args.weight_decay)
mlflow.log_param('eval_strategy', training_args.eval_strategy)
mlflow.log_param('save_strategy', training_args.save_strategy)
mlflow.log_param('logging_steps', training_args.logging_steps)

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
mlflow.log_metrics(eval_results)

# Classification report
preds = trainer.predict(val_ds)
pred_labels = np.argmax(preds.predictions, axis=1)
report = classification_report(val_ds['label'], pred_labels, target_names=labels)
print(report)
# Log classification report as artifact
import os
os.makedirs('results', exist_ok=True)
report_path = 'results/transformer_classification_report.txt'
with open(report_path, 'w') as f:
    f.write(report)
mlflow.log_artifact(report_path)

# Log model artifact using PyTorch flavor (needed for registration)
import mlflow.pytorch
from mlflow.models.signature import infer_signature

# Prepare sample for signature inference
example_input = {
    'input_ids': train_ds[0]['input_ids'].tolist(),
    'attention_mask': train_ds[0]['attention_mask'].tolist()
}
# Run raw model to get example output - ensure tensors are on same device as model
with torch.no_grad():
    import torch as _torch
    device = next(model.parameters()).device  # Get the device the model is on
    input_ids_tensor = _torch.tensor([example_input['input_ids']]).to(device)
    attention_tensor = _torch.tensor([example_input['attention_mask']]).to(device)
    output = model(input_ids=input_ids_tensor, attention_mask=attention_tensor).logits.cpu().numpy()
signature = infer_signature(example_input, output)

# Log and register model with signature and input example
mlflow.pytorch.log_model(
    model,
    artifact_path='transformer_model',
    signature=signature,
    input_example=example_input,
    registered_model_name='transformer_model'
)
  
# End MLflow run
mlflow.end_run()
