import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
import mlflow
from mlflow.tracking import MlflowClient
import os

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
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')

print("Train:", train_df.shape, "Val:", val_df.shape)

# 2) Stage-1: train a domain detector
domain_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8)),
    ('clf',   LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])
domain_pipe.fit(train_df['text'], train_df['domain'])
dom_val_preds = domain_pipe.predict(val_df['text'])
print("Domain Accuracy:", accuracy_score(val_df['domain'], dom_val_preds))

# 3) Stage-2: train your topic classifier using the *true* domain on train
feature_union = ColumnTransformer([
    ('text_tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8), 'text'),
    ('dom_ohe',    OneHotEncoder(handle_unknown='ignore'), ['domain'])
], sparse_threshold=0)

# Define parameter distributions for each model
param_dists = {
    "LogisticRegression": {
        'features__text_tfidf__ngram_range': [(1,1), (1,2)],
        'features__text_tfidf__min_df': [1, 3, 5],
        'clf__C': [0.01, 0.1, 1, 10, 100],
    },
    "LinearSVC": {
        'features__text_tfidf__ngram_range': [(1,1), (1,2)],
        'features__text_tfidf__min_df': [1, 3, 5],
        'clf__C': [0.01, 0.1, 1, 10, 100],
    },
    "MultinomialNB": {
        'features__text_tfidf__ngram_range': [(1,1), (1,2)],
        'features__text_tfidf__min_df': [1, 3, 5],
        'clf__alpha': [0.01, 0.1, 1, 5, 10],
    },
}

results = {}
for name, clf in [
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    ("LinearSVC", LinearSVC(max_iter=1000, random_state=42)),
    ("MultinomialNB", MultinomialNB()),
]:
    topic_pipe = Pipeline([
        ('features', feature_union),
        ('clf', clf)
    ])
    print(f"\nRunning RandomizedSearchCV for {name}...")
    rand = RandomizedSearchCV(topic_pipe, param_dists[name], n_iter=10, cv=3, n_jobs=4, scoring='accuracy', verbose=1, random_state=42)
    rand.fit(train_df[['text','domain']], train_df['target'])
    best_model = rand.best_estimator_
    val_df['domain'] = dom_val_preds  # use predicted domain
    preds = best_model.predict(val_df[['text','domain']])
    acc = accuracy_score(val_df['target'], preds)
    report = classification_report(val_df['target'], preds, output_dict=True)
    results[name] = {'accuracy': acc, 'report': report, 'best_params': rand.best_params_}
    print(f"\n{name} Validation Results (Best Params: {rand.best_params_}):")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(val_df['target'], preds))

print("\nSummary Table:")
for name, res in results.items():
    print(f"{name:18} Accuracy: {res['accuracy']:.4f} | Best Params: {res['best_params']}")

# Results:
#     LogisticRegression Accuracy: 0.8791 | Best Params: {'features__text_tfidf__ngram_range': (1, 2), 'features__text_tfidf__min_df': 3, 'clf__C': 100}
# LinearSVC          Accuracy: 0.8799 | Best Params: {'features__text_tfidf__ngram_range': (1, 2), 'features__text_tfidf__min_df': 3, 'clf__C': 1}
# MultinomialNB      Accuracy: 0.8669 | Best Params: {'features__text_tfidf__ngram_range': (1, 1), 'features__text_tfidf__min_df': 1, 'clf__alpha': 0.01}


