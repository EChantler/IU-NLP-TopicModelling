import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import os
import optuna
import gc
import tempfile
import shutil

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 5e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_epochs = trial.suggest_int('num_epochs', 2, 5)
    weight_decay = trial.suggest_float('weight_decay', 0.005, 0.05, log=True)
    max_length = trial.suggest_categorical('max_length', [128, 256, 512])
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.2)
    
    # Start MLflow run for this trial
    with mlflow.start_run(nested=True):
        try:
            # Log trial parameters
            mlflow.log_param('learning_rate', learning_rate)
            mlflow.log_param('batch_size', batch_size)
            mlflow.log_param('num_epochs', num_epochs)
            mlflow.log_param('weight_decay', weight_decay)
            mlflow.log_param('max_length', max_length)
            mlflow.log_param('warmup_ratio', warmup_ratio)
            mlflow.log_param('trial_number', trial.number)
            
            # Load data (same as in 7_1_train_transformer.py)
            client = MlflowClient()
            experiments = client.search_experiments()
            data_run = None
            
            for experiment in experiments:
                runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=10)
                for run in runs:
                    try:
                        artifacts = client.list_artifacts(run.info.run_id, path="cleaned_data")
                        if artifacts:
                            data_run = run
                            break
                    except Exception:
                        continue
                if data_run:
                    break
            
            if not data_run:
                raise FileNotFoundError("No runs with cleaned_data artifacts found")
            
            # Download data artifacts
            data_path = mlflow.artifacts.download_artifacts(artifact_path="cleaned_data", run_id=data_run.info.run_id)
            
            # Load transformer processed data
            train_df = pd.read_csv(os.path.join(data_path, 'train_transformer.csv'))
            val_df = pd.read_csv(os.path.join(data_path, 'val_transformer.csv'))
            
            # # Prepare data with domain information
            # train_df['input_text'] = train_df['domain'] + ': ' + train_df['text']
            # val_df['input_text'] = val_df['domain'] + ': ' + val_df['text']
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
            
            # Create datasets
            train_ds = Dataset.from_pandas(train_df[['input_text', 'label']])
            val_ds = Dataset.from_pandas(val_df[['input_text', 'label']])
            
            # Model configuration
            model_name = 'distilbert-base-uncased'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            def tokenize_function(batch):
                return tokenizer(
                    batch['input_text'], 
                    truncation=True, 
                    padding='max_length', 
                    max_length=max_length
                )
            
            # Tokenize datasets
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
            
            # Create temporary directory for this trial
            temp_dir = tempfile.mkdtemp(prefix=f'trial_{trial.number}_')
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=temp_dir,
                eval_strategy='epoch',
                save_strategy='epoch',
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                logging_dir=None,
                logging_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model='f1_macro',
                greater_is_better=True,
                save_total_limit=1,  # Save space
                report_to=None,
                remove_unused_columns=True,
                dataloader_pin_memory=False,  # Save memory
            )
            
            def compute_metrics(eval_pred):
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
            trainer.train()
            
            # Evaluate on validation set
            eval_results = trainer.evaluate()
            
            # Get the metrics we care about
            val_f1_macro = eval_results.get('eval_f1_macro', 0.0)
            val_accuracy = eval_results.get('eval_accuracy', 0.0)
            
            # Log metrics
            mlflow.log_metric('val_f1_macro', val_f1_macro)
            mlflow.log_metric('val_accuracy', val_accuracy)
            
            print(f"Trial {trial.number}: F1={val_f1_macro:.4f}, Acc={val_accuracy:.4f}")
            
            # Clean up
            del model
            del trainer
            del train_ds
            del val_ds
            torch.cuda.empty_cache()
            gc.collect()
            
            # Remove temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            return val_f1_macro  # Optimize for F1-macro score
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            # Clean up on failure
            torch.cuda.empty_cache()
            gc.collect()
            return 0.0  # Return poor score for failed trials

def main():
    # Set MLflow experiment
    experiment_name = "transformer_hyperparameter_search"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="hyperparameter_search_session"):
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',  # Maximize F1-macro score
            study_name='transformer_hyperparam_search',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
        )
        
        # Log study configuration
        mlflow.log_param('optimization_direction', 'maximize')
        mlflow.log_param('target_metric', 'f1_macro')
        mlflow.log_param('sampler', 'TPESampler')
        mlflow.log_param('pruner', 'MedianPruner')
        
        # Run optimization
        n_trials = 20  # Adjust based on computational budget
        mlflow.log_param('n_trials', n_trials)
        
        print(f"Starting hyperparameter search with {n_trials} trials...")
        print("Optimizing for F1-macro score on validation set")
        print("=" * 60)
        
        study.optimize(objective, n_trials=n_trials, timeout=None)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        print("\n" + "=" * 60)
        print("HYPERPARAMETER SEARCH RESULTS")
        print("=" * 60)
        print(f"Best F1-macro score: {best_score:.4f}")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Log best results
        mlflow.log_metric('best_f1_macro', best_score)
        for param, value in best_params.items():
            mlflow.log_param(f'best_{param}', value)
        
        # Save study results
        study_df = study.trials_dataframe()
        study_csv_path = './results/hyperparameter_search_results.csv'
        os.makedirs('./results', exist_ok=True)
        study_df.to_csv(study_csv_path, index=False)
        mlflow.log_artifact(study_csv_path)
        
        # Create a summary report
        report_path = './results/hyperparameter_search_summary.txt'
        with open(report_path, 'w') as f:
            f.write("TRANSFORMER HYPERPARAMETER SEARCH SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write(f"Best F1-macro score: {best_score:.4f}\n")
            f.write(f"Best trial number: {study.best_trial.number}\n\n")
            f.write("Best hyperparameters:\n")
            for param, value in best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            f.write("Top 5 trials:\n")
            top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
            for i, trial in enumerate(top_trials, 1):
                f.write(f"  {i}. Trial {trial.number}: F1={trial.value:.4f} ")
                f.write(f"(lr={trial.params.get('learning_rate', 'N/A'):.2e}, ")
                f.write(f"bs={trial.params.get('batch_size', 'N/A')}, ")
                f.write(f"epochs={trial.params.get('num_epochs', 'N/A')})\n")
        
        mlflow.log_artifact(report_path)
        
        print(f"\nResults saved to {study_csv_path}")
        print(f"Summary saved to {report_path}")
        print("\nNext steps:")
        print("1. Use the best parameters to train your final model")
        print("2. Consider running more trials if computational budget allows")
        print("3. Check MLflow UI for detailed trial comparisons")

if __name__ == "__main__":
    main()

# # OUTPUT:
# [I 2025-08-10 20:49:00,544] Trial 19 finished with value: 0.6341323973876628 and parameters: {'learning_rate': 3.7571180388481106e-05, 'batch_size': 8, 'num_epochs': 4, 'weight_decay': 0.014017620414272097, 'max_length': 512, 'warmup_ratio': 0.16436782727198565}. Best is trial 15 with value: 0.6434687598847755.

# ============================================================
# HYPERPARAMETER SEARCH RESULTS
# ============================================================
# Best F1-macro score: 0.6435
# Best parameters:
#   learning_rate: 3.020000175076628e-05
#   batch_size: 8
#   num_epochs: 5
#   weight_decay: 0.005792313337270436
#   max_length: 512
#   warmup_ratio: 0.1603350974360887

# Results saved to ./results/hyperparameter_search_results.csv
# Summary saved to ./results/hyperparameter_search_summary.txt