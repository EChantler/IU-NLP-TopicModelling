import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import optuna
import gc
import numpy as np

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters for TF-IDF
    ngram_min = trial.suggest_int('ngram_min', 1, 2)
    ngram_max = trial.suggest_int('ngram_max', ngram_min, 3)  # Ensure max >= min
    min_df = trial.suggest_int('min_df', 2, 10)
    max_df = trial.suggest_float('max_df', 0.5, 0.95)
    max_features = trial.suggest_categorical('max_features', [None, 5000, 10000, 20000, 50000])
    
    # Suggest hyperparameters for Logistic Regression
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    max_iter = trial.suggest_categorical('max_iter', [500, 1000, 2000])
    solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
    l1_ratio = None
    penalty = 'l2'  # Default for most solvers
    
    # For liblinear, we can try different penalties
    if solver == 'liblinear':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    elif solver == 'lbfgs':
        # lbfgs only supports l2 penalty
        penalty = 'l2'
    
    # Start MLflow run for this trial
    with mlflow.start_run(nested=True):
        try:
            # Log trial parameters
            mlflow.log_param('ngram_range', f'({ngram_min}, {ngram_max})')
            mlflow.log_param('min_df', min_df)
            mlflow.log_param('max_df', max_df)
            mlflow.log_param('max_features', max_features)
            mlflow.log_param('C', C)
            mlflow.log_param('max_iter', max_iter)
            mlflow.log_param('solver', solver)
            mlflow.log_param('penalty', penalty)
            mlflow.log_param('class_weight', class_weight)
            mlflow.log_param('trial_number', trial.number)
            
            # Load data (same as in 6_train_classic.py)
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
            
            # Load TF-IDF processed data (aggressive cleaning for classical ML)
            train_df = pd.read_csv(os.path.join(data_path, 'train_tfidf.csv'))
            val_df = pd.read_csv(os.path.join(data_path, 'val_tfidf.csv'))
            
            # Create and configure the pipeline
            pipeline_params = {
                'tfidf__ngram_range': (ngram_min, ngram_max),
                'tfidf__min_df': min_df,
                'tfidf__max_df': max_df,
                'tfidf__max_features': max_features,
                'clf__C': C,
                'clf__max_iter': max_iter,
                'clf__solver': solver,
                'clf__penalty': penalty,
                'clf__class_weight': class_weight,
                'clf__random_state': 42
            }
            
            target_pipe = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression())
            ])
            
            # Set parameters
            target_pipe.set_params(**pipeline_params)
            
            # Train the model
            target_pipe.fit(train_df['text'], train_df['target'])
            
            # Evaluate on validation set
            preds_val = target_pipe.predict(val_df['text'])
            acc_val = accuracy_score(val_df['target'], preds_val)
            f1_val = f1_score(val_df['target'], preds_val, average='macro', zero_division=0)
            
            # Log metrics
            mlflow.log_metric('val_accuracy', acc_val)
            mlflow.log_metric('val_f1_macro', f1_val)
            
            print(f"Trial {trial.number}: F1={f1_val:.4f}, Acc={acc_val:.4f}")
            
            # Clean up
            del target_pipe
            gc.collect()
            
            return f1_val  # Optimize for F1-macro score
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            gc.collect()
            return 0.0  # Return poor score for failed trials

def main():
    # Set MLflow experiment
    experiment_name = "classic_ml_hyperparameter_search"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="logistic_regression_search_session"):
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',  # Maximize F1-macro score
            study_name='logistic_regression_hyperparam_search',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        )
        
        # Log study configuration
        mlflow.log_param('optimization_direction', 'maximize')
        mlflow.log_param('target_metric', 'f1_macro')
        mlflow.log_param('sampler', 'TPESampler')
        mlflow.log_param('pruner', 'MedianPruner')
        mlflow.log_param('model_type', 'TF-IDF + Logistic Regression')
        
        # Run optimization
        n_trials = 50  # More trials for classical ML since they're faster
        mlflow.log_param('n_trials', n_trials)
        
        print(f"Starting hyperparameter search with {n_trials} trials...")
        print("Optimizing TF-IDF + Logistic Regression pipeline")
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
        print("\nBest parameters:")
        print("TF-IDF Vectorizer:")
        print(f"  ngram_range: ({best_params['ngram_min']}, {best_params['ngram_max']})")
        print(f"  min_df: {best_params['min_df']}")
        print(f"  max_df: {best_params['max_df']}")
        print(f"  max_features: {best_params['max_features']}")
        print("\nLogistic Regression:")
        print(f"  C: {best_params['C']}")
        print(f"  max_iter: {best_params['max_iter']}")
        print(f"  solver: {best_params['solver']}")
        # Handle penalty parameter - it might not exist for lbfgs solver
        penalty = best_params.get('penalty', 'l2')  # Default to l2 if not present
        print(f"  penalty: {penalty}")
        print(f"  class_weight: {best_params['class_weight']}")
        
        # Log best results
        mlflow.log_metric('best_f1_macro', best_score)
        for param, value in best_params.items():
            mlflow.log_param(f'best_{param}', value)
        
        # Save study results
        study_df = study.trials_dataframe()
        study_csv_path = './results/classic_ml_hyperparameter_search_results.csv'
        os.makedirs('./results', exist_ok=True)
        study_df.to_csv(study_csv_path, index=False)
        mlflow.log_artifact(study_csv_path)
        
        # Train final model with best parameters and save detailed results
        print("\n" + "=" * 60)
        print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print("=" * 60)
        
        # Load data again for final training
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
        
        if data_run:
            data_path = mlflow.artifacts.download_artifacts(artifact_path="cleaned_data", run_id=data_run.info.run_id)
            train_df = pd.read_csv(os.path.join(data_path, 'train_tfidf.csv'))
            val_df = pd.read_csv(os.path.join(data_path, 'val_tfidf.csv'))
            
            # Create final pipeline with best parameters
            penalty = best_params.get('penalty', 'l2')  # Default to l2 if not present
            final_pipe = Pipeline([
                ('tfidf', TfidfVectorizer(
                    ngram_range=(best_params['ngram_min'], best_params['ngram_max']),
                    min_df=best_params['min_df'],
                    max_df=best_params['max_df'],
                    max_features=best_params['max_features']
                )),
                ('clf', LogisticRegression(
                    C=best_params['C'],
                    max_iter=best_params['max_iter'],
                    solver=best_params['solver'],
                    penalty=penalty,
                    class_weight=best_params['class_weight'],
                    random_state=42
                ))
            ])
            
            # Train final model
            final_pipe.fit(train_df['text'], train_df['target'])
            
            # Evaluate final model
            train_preds = final_pipe.predict(train_df['text'])
            val_preds = final_pipe.predict(val_df['text'])
            
            train_acc = accuracy_score(train_df['target'], train_preds)
            train_f1 = f1_score(train_df['target'], train_preds, average='macro', zero_division=0)
            val_acc = accuracy_score(val_df['target'], val_preds)
            val_f1 = f1_score(val_df['target'], val_preds, average='macro', zero_division=0)
            
            print(f"Final model performance:")
            print(f"  Training: Accuracy={train_acc:.4f}, F1-macro={train_f1:.4f}")
            print(f"  Validation: Accuracy={val_acc:.4f}, F1-macro={val_f1:.4f}")
            
            # Log final metrics
            mlflow.log_metric('final_train_accuracy', train_acc)
            mlflow.log_metric('final_train_f1_macro', train_f1)
            mlflow.log_metric('final_val_accuracy', val_acc)
            mlflow.log_metric('final_val_f1_macro', val_f1)
            
            # Save final model
            final_model_path = './results/optimized_classic_model.joblib'
            joblib.dump(final_pipe, final_model_path)
            mlflow.log_artifact(final_model_path)
            
            # Generate detailed classification report
            val_report = classification_report(
                val_df['target'], 
                val_preds, 
                digits=4,
                zero_division=0
            )
            
            print("\nDetailed Classification Report (Validation Set):")
            print(val_report)
        
        # Create a comprehensive summary report
        report_path = './results/classic_ml_hyperparameter_search_summary.txt'
        with open(report_path, 'w') as f:
            f.write("CLASSIC ML HYPERPARAMETER SEARCH SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: TF-IDF + Logistic Regression\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write(f"Best F1-macro score: {best_score:.4f}\n")
            f.write(f"Best trial number: {study.best_trial.number}\n\n")
            
            f.write("Best hyperparameters:\n")
            f.write("TF-IDF Vectorizer:\n")
            f.write(f"  ngram_range: ({best_params['ngram_min']}, {best_params['ngram_max']})\n")
            f.write(f"  min_df: {best_params['min_df']}\n")
            f.write(f"  max_df: {best_params['max_df']}\n")
            f.write(f"  max_features: {best_params['max_features']}\n")
            f.write("Logistic Regression:\n")
            f.write(f"  C: {best_params['C']}\n")
            f.write(f"  max_iter: {best_params['max_iter']}\n")
            f.write(f"  solver: {best_params['solver']}\n")
            penalty = best_params.get('penalty', 'l2')  # Default to l2 if not present
            f.write(f"  penalty: {penalty}\n")
            f.write(f"  class_weight: {best_params['class_weight']}\n\n")
            
            f.write("Top 5 trials:\n")
            top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
            for i, trial in enumerate(top_trials, 1):
                f.write(f"  {i}. Trial {trial.number}: F1={trial.value:.4f} ")
                f.write(f"(C={trial.params.get('C', 'N/A'):.3f}, ")
                f.write(f"ngram=({trial.params.get('ngram_min', 'N/A')},{trial.params.get('ngram_max', 'N/A')}), ")
                f.write(f"solver={trial.params.get('solver', 'N/A')})\n")
            
            f.write(f"\nFinal optimized model performance:\n")
            if 'final_val_f1_macro' in locals():
                f.write(f"  Training: Accuracy={train_acc:.4f}, F1-macro={train_f1:.4f}\n")
                f.write(f"  Validation: Accuracy={val_acc:.4f}, F1-macro={val_f1:.4f}\n")
        
        mlflow.log_artifact(report_path)
        
        print(f"\nResults saved to {study_csv_path}")
        print(f"Summary saved to {report_path}")
        print(f"Optimized model saved to {final_model_path}")
        print("\nNext steps:")
        print("1. Use the best parameters to update your 6_train_classic.py script")
        print("2. Compare with transformer hyperparameter search results")
        print("3. Check MLflow UI for detailed trial comparisons")

if __name__ == "__main__":
    main()

## Output:
# [I 2025-08-10 18:33:19,899] Trial 49 finished with value: 0.5609616954824892 and parameters: {'ngram_min': 1, 'ngram_max': 1, 'min_df': 3, 'max_df': 0.5866814138568062, 'max_features': 20000, 'C': 0.13415439274161256, 'max_iter': 500, 'solver': 'lbfgs', 'class_weight': 'balanced'}. Best is trial 33 with value: 0.6071975594549611.

# ============================================================
# HYPERPARAMETER SEARCH RESULTS
# ============================================================
# Best F1-macro score: 0.6072

# Best parameters:
# TF-IDF Vectorizer:
#   ngram_range: (1, 2)
#   min_df: 3
#   max_df: 0.5257892626851385
#   max_features: 20000

# Logistic Regression:
#   C: 0.9074963597476341
#   max_iter: 500
#   solver: lbfgs
#   penalty: l2
#   class_weight: balanced

# ============================================================
# TRAINING FINAL MODEL WITH BEST PARAMETERS
# ============================================================
# Downloading artifacts: 100%|███████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 782.63it/s]
# Final model performance:
#   Training: Accuracy=0.9155, F1-macro=0.9317
#   Validation: Accuracy=0.6778, F1-macro=0.6072

# Detailed Classification Report (Validation Set):
#                                                     precision    recall  f1-score   support

#                        Arts, Media & Entertainment     0.6786    0.6552    0.6667        58
#                  Business, Work & Personal Finance     0.6250    0.6000    0.6122        25
#                               Computing, AI & Data     0.7097    0.6197    0.6617        71
#                        Culture, Lifestyle & Travel     0.5294    0.4615    0.4932        39
#                       Earth, Environment & Climate     0.0000    0.0000    0.0000         2
#                       Economics, Finance & Markets     0.3636    0.5714    0.4444         7
#     Engineering (Electrical, Mechanical & Systems)     0.2143    0.1667    0.1875        18
#                                 Hardware & Devices     0.4444    0.8571    0.5854        14
#              International Relations & Geopolitics     0.7000    0.7778    0.7368         9
#                      Life Sciences & Biotechnology     0.3636    0.5000    0.4211         8
#                          Mathematics & Foundations     0.6909    0.7917    0.7379        48
#                           Medicine & Public Health     0.9130    0.8750    0.8936        24
# Physical Sciences (Physics, Chemistry & Materials)     0.8415    0.7667    0.8023        90
#                      Politics, Law & Public Policy     0.4750    0.6786    0.5588        28
#                                  Religion & Belief     0.8125    0.5909    0.6842        22
#                            Security & Cryptography     0.9000    0.7500    0.8182        12
#                       Social & Behavioral Sciences     0.6667    0.4000    0.5000         5
#                                  Space & Astronomy     0.8438    0.7941    0.8182        34
#                                             Sports     0.6765    0.7931    0.7302        29
#                          Transportation & Mobility     0.8261    0.7600    0.7917        25

#                                           accuracy                         0.6778       568
#                                          macro avg     0.6137    0.6205    0.6072       568
#                                       weighted avg     0.6918    0.6778    0.6791       568