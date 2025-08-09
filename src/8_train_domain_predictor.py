import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import joblib
import os
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Constants for model hyperparameters
NGRAM_RANGE = (1, 2)
MIN_DF = 5
MAX_DF = 0.8
MAX_ITER = 1000
RANDOM_STATE = 42
CLASS_WEIGHT = 'balanced'

mlflow.start_run()

if __name__ == "__main__":
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
            try:
                val_df = pd.read_csv(os.path.join(data_path, 'val.csv'))
                val_path_exists = True
            except FileNotFoundError:
                val_path_exists = False
            print(f"Loaded data from MLflow artifacts")
        else:
            raise FileNotFoundError("No runs with data_splits artifacts found in any experiment")
            
    except Exception as e:
        print(f"Failed to load data from MLflow: {e}")
        print("Falling back to local data files")
        # Fallback to local files
        train_df = pd.read_csv('data/processed/train.csv')
        val_path = 'data/processed/val.csv'
        val_path_exists = os.path.exists(val_path)
        if val_path_exists:
            val_df = pd.read_csv(val_path)
    
    domain_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF, max_df=MAX_DF)),
        ('clf',   LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE, class_weight=CLASS_WEIGHT))
    ])
    # Log model parameters
    mlflow.log_param('ngram_range', NGRAM_RANGE)
    mlflow.log_param('min_df', MIN_DF)
    mlflow.log_param('max_df', MAX_DF)
    mlflow.log_param('max_iter', MAX_ITER)
    mlflow.log_param('random_state', RANDOM_STATE)
    mlflow.log_param('class_weight', CLASS_WEIGHT)
    domain_pipe.fit(train_df['text'], train_df['domain'])
    # Validation on train set
    preds_train = domain_pipe.predict(train_df['text'])
    acc_train = accuracy_score(train_df['domain'], preds_train)
    print(f"Domain predictor accuracy (train set): {acc_train:.4f}")
    mlflow.log_metric('accuracy_train', acc_train)
    print("Classification report (train set):")
    print(classification_report(train_df['domain'], preds_train))

    # Validation on val set
    if val_path_exists:
        preds_val = domain_pipe.predict(val_df['text'])
        acc_val = accuracy_score(val_df['domain'], preds_val)
        print(f"\nDomain predictor accuracy (val set): {acc_val:.4f}")
        mlflow.log_metric('accuracy_val', acc_val)
        print("Classification report (val set):")
        print(classification_report(val_df['domain'], preds_val))
    else:
        print("Validation file not found. Skipping val set evaluation.")

    os.makedirs('./results', exist_ok=True)
    joblib.dump(domain_pipe, './results/domain_pipe.joblib')
    # Prepare input example and infer model signature
    input_example = train_df[['text']].head(3)
    signature = infer_signature(input_example, domain_pipe.predict(input_example['text']))
    # Log model artifact with name, signature, and input example
    mlflow.sklearn.log_model(domain_pipe, name='domain_model', signature=signature, input_example=input_example)
    print('Domain predictor trained and saved to ./results/domain_pipe.joblib')
    mlflow.end_run()
