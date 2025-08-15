import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import os

RUN_DESCRIPTION = ""

mlflow.start_run()

if __name__ == "__main__":
    # Load data from MLflow artifacts
    client = MlflowClient()
    try:
        # Get all experiments and search for cleaned_data artifacts
        experiments = client.search_experiments()
        data_run = None
        
        for experiment in experiments:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
            for run in runs:
                try:
                    artifacts = client.list_artifacts(run.info.run_id, path="cleaned_data")
                    if artifacts:  # Found a run with cleaned_data
                        data_run = run
                        print(f"Found cleaned_data in experiment {experiment.name} (ID: {experiment.experiment_id})")
                        break
                except Exception:
                    continue
            if data_run:
                break
        
        if data_run:
            print(f"Loading data from MLflow run: {data_run.info.run_id}")
            # Download data artifacts
            data_path = mlflow.artifacts.download_artifacts(artifact_path="cleaned_data", run_id=data_run.info.run_id)
            
            # Load transformer processed data (light cleaning for transformers)
            train_df = pd.read_csv(os.path.join(data_path, 'train_transformer.csv'))
            val_df = pd.read_csv(os.path.join(data_path, 'val_transformer.csv'))
            # Note: test_df is intentionally not loaded - reserved for final evaluation
            
            print(f"Loaded transformer data from MLflow artifacts")
            print(f"Train set: {len(train_df)} samples")
            print(f"Validation set: {len(val_df)} samples")
            print(f"Target classes: {train_df['target'].nunique()}")
            print(f"Domains: {train_df['domain'].nunique()}")
            
        else:
            raise FileNotFoundError("No runs with cleaned_data artifacts found in any experiment")
            
    except Exception as e:
        print(f"Failed to load data from MLflow: {e}")
        raise RuntimeError(f"Failed to load data from registry: {e}")
    
    # Log dataset information
    mlflow.log_param('dataset_source', 'transformer_cleaned')
    mlflow.log_param('train_samples', len(train_df))
    mlflow.log_param('val_samples', len(val_df))
    mlflow.log_param('num_classes', train_df['target'].nunique())
    mlflow.log_param('num_domains', train_df['domain'].nunique())
    
    # Train transformer WITHOUT domain information to avoid data leakage
    # Use only the text content for classification
    train_df['input_text'] = train_df['text']
    val_df['input_text'] = val_df['text']
    
    # Encode labels
    labels = sorted(train_df['target'].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    train_df['label'] = train_df['target'].map(label2id)
    val_df['label'] = val_df['target'].map(label2id)
    
    print(f"Label mapping: {label2id}")
    
    # Create Hugging Face datasets
    train_ds = Dataset.from_pandas(train_df[['input_text', 'label']])
    val_ds = Dataset.from_pandas(val_df[['input_text', 'label']])
    
    # Model & Tokenizer configuration (optimized parameters)
    model_name = 'distilbert-base-uncased'
    max_length = 256
    learning_rate = 4.769422631212765e-05
    batch_size = 16
    num_epochs = 4
    weight_decay = 0.04508605470151595
    warmup_ratio = 0.13804941471382906
    
    # Log model parameters
    mlflow.log_param('model_name', model_name)
    mlflow.log_param('max_length', max_length)
    mlflow.log_param('include_domain', False)
    mlflow.log_param('hyperparameter_optimized', True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(batch):
        return tokenizer(
            batch['input_text'], 
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    cols = ['input_ids', 'attention_mask', 'label']
    train_ds.set_format(type='torch', columns=cols)
    val_ds.set_format(type='torch', columns=cols)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(labels), 
        id2label=id2label, 
        label2id=label2id
    )
    
    # Training arguments (optimized parameters)
    training_args = TrainingArguments(
        output_dir='./results/checkpoints',
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        save_total_limit=2,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Log training hyperparameters
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('train_batch_size', batch_size)
    mlflow.log_param('eval_batch_size', batch_size)
    mlflow.log_param('num_train_epochs', num_epochs)
    mlflow.log_param('weight_decay', weight_decay)
    mlflow.log_param('warmup_ratio', warmup_ratio)
    mlflow.log_param('eval_strategy', training_args.eval_strategy)
    mlflow.log_param('save_strategy', training_args.save_strategy)
    mlflow.log_param('logging_steps', training_args.logging_steps)
    mlflow.log_param('metric_for_best_model', 'f1_macro')
    mlflow.log_param('run_description', RUN_DESCRIPTION)
    
    def compute_metrics(eval_pred):
        from sklearn.metrics import f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = (preds == labels).mean()
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        return {
            'accuracy': acc,
            'f1_macro': f1_macro
        }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save the model and tokenizer locally
    model_save_path = './results/final_transformer_model'
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")
    
    # Evaluate on training set (for comparison with classical ML)
    print("\n=== Training Set Evaluation ===")
    train_preds = trainer.predict(train_ds)
    train_pred_labels = np.argmax(train_preds.predictions, axis=1)
    train_true_labels = train_ds['label'].numpy()
    
    from sklearn.metrics import f1_score
    acc_train = (train_pred_labels == train_true_labels).mean()
    f1_macro_train = f1_score(train_true_labels, train_pred_labels, average='macro', zero_division=0)
    
    print(f"Transformer accuracy (train set): {acc_train:.4f}")
    print(f"Transformer F1-macro (train set): {f1_macro_train:.4f}")
    mlflow.log_metric('accuracy_train', acc_train)
    mlflow.log_metric('f1_train', f1_macro_train)
    
    train_report = classification_report(
        train_true_labels, 
        train_pred_labels, 
        target_names=labels,
        digits=4,
        zero_division=0
    )
    print("Classification report (train set):")
    print(train_report)
    
    # Evaluate on validation set
    print("\n=== Validation Set Evaluation ===")
    eval_results = trainer.evaluate()
    preds = trainer.predict(val_ds)
    pred_labels = np.argmax(preds.predictions, axis=1)
    true_labels = val_ds['label'].numpy()
    
    acc_val = (pred_labels == true_labels).mean()
    f1_macro_val = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    
    print(f"Transformer accuracy (validation set): {acc_val:.4f}")
    print(f"Transformer F1-macro (validation set): {f1_macro_val:.4f}")
    mlflow.log_metric('accuracy_val', acc_val)
    mlflow.log_metric('f1_val', f1_macro_val)
    
    print('Validation Results:', eval_results)
    
    # Generate detailed classification report for validation set
    print("\nGenerating classification report...")
    val_report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=labels,
        digits=4,
        zero_division=0
    )
    print("\nClassification Report (Validation Set):")
    print(val_report)
    
    # Save classification reports
    os.makedirs('./results', exist_ok=True)
    
    # Save training classification report
    train_report_path = './results/transformer_classification_report_train.txt'
    with open(train_report_path, 'w') as f:
        f.write("TRANSFORMER MODEL CLASSIFICATION REPORT (TRAINING SET)\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: transformer_cleaned (text only, no domain)\n")
        f.write(f"Training Accuracy: {acc_train:.4f}\n\n")
        f.write(train_report)
    
    # Save validation classification report
    val_report_path = './results/transformer_classification_report_val.txt'
    with open(val_report_path, 'w') as f:
        f.write("TRANSFORMER MODEL CLASSIFICATION REPORT (VALIDATION SET)\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: transformer_cleaned (text only, no domain)\n")
        f.write(f"Validation Accuracy: {acc_val:.4f}\n\n")
        f.write(val_report)
    
    # Log classification reports as artifacts
    mlflow.log_artifact(train_report_path)
    mlflow.log_artifact(val_report_path)
    
    # Prepare input example and signature for model logging
    # Use a simpler format that MLflow can handle better
    example_input_ids = train_ds[0]['input_ids'].tolist()
    example_attention_mask = train_ds[0]['attention_mask'].tolist()
    
    # Generate example output for signature
    with torch.no_grad():
        device = next(model.parameters()).device
        input_ids_tensor = torch.tensor([example_input_ids]).to(device)
        attention_tensor = torch.tensor([example_attention_mask]).to(device)
        output = model(input_ids=input_ids_tensor, attention_mask=attention_tensor).logits.cpu().numpy()
    
    # Create a simpler input example that avoids nested structures
    serving_input_example = {
        'input_ids': example_input_ids,
        'attention_mask': example_attention_mask
    }
    
    signature = infer_signature(serving_input_example, output)
    
    # Log model to MLflow without input_example to avoid validation issues
    mlflow.pytorch.log_model(
        model,
        artifact_path='transformer_model',
        signature=signature,
        registered_model_name='transformer_model'
    )
    
    # Log additional artifacts
    mlflow.log_artifacts('./results', artifact_path="model_artifacts")
    
    print(f'\nModel logged to MLflow with accuracy: Train={acc_train:.4f}, Val={acc_val:.4f}')
    print("Note: Test set evaluation reserved for final model assessment")
    
    # Show some example predictions
    print(f"\n=== Example Predictions (Validation Set) ===")
    sample_indices = np.random.choice(len(val_df), size=5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample_text = val_df.iloc[idx]['input_text']
        true_target = val_df.iloc[idx]['target']
        true_label = val_df.iloc[idx]['label']
        
        # Get prediction for this sample
        pred_label = pred_labels[idx]
        pred_target = id2label[pred_label]
        
        print(f"\nExample {i+1}:")
        print(f"Text (first 100 chars): {sample_text[:100]}...")
        print(f"True target: {true_target}")
        print(f"Predicted target: {pred_target}")
        print(f"Correct: {'✓' if true_target == pred_target else '✗'}")

# End MLflow run
mlflow.end_run()
