import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

RUN_DESCRIPTION = ""

# Constants for model hyperparameters
NGRAM_RANGE = (1, 2)
MIN_DF = 3
MAX_DF = 0.5257892626851385
MAX_FEATURES = 20000
C = 0.9074963597476341
MAX_ITER = 500
SOLVER = 'lbfgs'
PENALTY = 'l2'
RANDOM_STATE = 42
CLASS_WEIGHT = 'balanced'

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
            
            # Load TF-IDF processed data (aggressive cleaning for classical ML)
            train_df = pd.read_csv(os.path.join(data_path, 'train_tfidf.csv'))
            val_df = pd.read_csv(os.path.join(data_path, 'val_tfidf.csv'))
            # Note: test_df is intentionally not loaded - reserved for final evaluation
            
            print(f"Loaded TF-IDF data from MLflow artifacts")
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
    mlflow.log_param('dataset_source', 'tfidf_cleaned')
    mlflow.log_param('train_samples', len(train_df))
    mlflow.log_param('val_samples', len(val_df))
    mlflow.log_param('num_classes', train_df['target'].nunique())
    mlflow.log_param('num_domains', train_df['domain'].nunique())
    
    # Create and configure the pipeline for target prediction
    target_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=NGRAM_RANGE, 
            min_df=MIN_DF, 
            max_df=MAX_DF,
            max_features=MAX_FEATURES
        )),
        ('clf', LogisticRegression(
            C=C,
            max_iter=MAX_ITER,
            solver=SOLVER,
            penalty=PENALTY,
            random_state=RANDOM_STATE, 
            class_weight=CLASS_WEIGHT
        ))
    ])
    
    # Log model hyperparameters
    mlflow.log_param('model_type', 'tfidf_logistic_regression')
    mlflow.log_param('target_prediction', True)
    mlflow.log_param('ngram_range', NGRAM_RANGE)
    mlflow.log_param('min_df', MIN_DF)
    mlflow.log_param('max_df', MAX_DF)
    mlflow.log_param('max_features', MAX_FEATURES)
    mlflow.log_param('C', C)
    mlflow.log_param('max_iter', MAX_ITER)
    mlflow.log_param('solver', SOLVER)
    mlflow.log_param('penalty', PENALTY)
    mlflow.log_param('random_state', RANDOM_STATE)
    mlflow.log_param('class_weight', CLASS_WEIGHT)
    mlflow.log_param('run_description', RUN_DESCRIPTION)
    mlflow.log_param('hyperparameter_optimized', True)
    
    # Train the model
    print("\nTraining target prediction model...")
    target_pipe.fit(train_df['text'], train_df['target'])
    
    # Evaluate on training set
    print("\n=== Training Set Evaluation ===")
    preds_train = target_pipe.predict(train_df['text'])
    acc_train = accuracy_score(train_df['target'], preds_train)
    
    from sklearn.metrics import f1_score
    f1_train = f1_score(train_df['target'], preds_train, average='macro', zero_division=0)
    
    print(f"Target predictor accuracy (train set): {acc_train:.4f}")
    print(f"Target predictor F1-macro (train set): {f1_train:.4f}")
    mlflow.log_metric('accuracy_train', acc_train)
    mlflow.log_metric('f1_train', f1_train)
    
    print("Classification report (train set):")
    train_report = classification_report(train_df['target'], preds_train)
    print(train_report)
    
    # Evaluate on validation set
    print("\n=== Validation Set Evaluation ===")
    preds_val = target_pipe.predict(val_df['text'])
    acc_val = accuracy_score(val_df['target'], preds_val)
    f1_val = f1_score(val_df['target'], preds_val, average='macro', zero_division=0)
    
    print(f"Target predictor accuracy (validation set): {acc_val:.4f}")
    print(f"Target predictor F1-macro (validation set): {f1_val:.4f}")
    mlflow.log_metric('accuracy_val', acc_val)
    mlflow.log_metric('f1_val', f1_val)
    
    print("Classification report (validation set):")
    val_report = classification_report(val_df['target'], preds_val)
    print(val_report)
    
    # Save classification reports
    os.makedirs('./results', exist_ok=True)
    with open('./results/target_classification_report_train.txt', 'w') as f:
        f.write("TRAINING SET CLASSIFICATION REPORT\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy: {acc_train:.4f}\n\n")
        f.write(train_report)
    
    with open('./results/target_classification_report_val.txt', 'w') as f:
        f.write("VALIDATION SET CLASSIFICATION REPORT\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy: {acc_val:.4f}\n\n")
        f.write(val_report)
    
    # Save the model locally
    joblib.dump(target_pipe, './results/target_pipe.joblib')
    print('\nTarget predictor saved to ./results/target_pipe.joblib')
    
    # Prepare input example and infer model signature for MLflow
    input_example = train_df[['text']].head(3)
    signature = infer_signature(input_example, target_pipe.predict(input_example['text']))
    
    # Log model to MLflow with name, signature, and input example
    mlflow.sklearn.log_model(
        target_pipe, 
        name='logistic_regression_target_predictor', 
        signature=signature, 
        input_example=input_example
    )
    
    # Log classification reports as artifacts
    mlflow.log_artifacts('./results', artifact_path="classification_reports")
    
    print(f'\nModel logged to MLflow with accuracy: Train={acc_train:.4f}, Val={acc_val:.4f}')
    print("Note: Test set evaluation reserved for final model assessment")
    
    # Show some example predictions from validation set
    print(f"\n=== Example Predictions (Validation Set) ===")
    sample_texts = val_df['text'].head(5).tolist()
    sample_targets = val_df['target'].head(5).tolist()
    sample_predictions = target_pipe.predict(sample_texts)
    
    for i, (text, true_target, pred_target) in enumerate(zip(sample_texts, sample_targets, sample_predictions)):
        print(f"\nExample {i+1}:")
        print(f"Text (first 100 chars): {text[:100]}...")
        print(f"True target: {true_target}")
        print(f"Predicted target: {pred_target}")
        print(f"Correct: {'✓' if true_target == pred_target else '✗'}")

mlflow.end_run()
