import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import joblib
import os
from sklearn.metrics import classification_report, accuracy_score

if __name__ == "__main__":
    train_df = pd.read_csv('data/processed/train.csv')
    domain_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8)),
        ('clf',   LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    domain_pipe.fit(train_df['text'], train_df['domain'])
    # Validation on train set
    preds_train = domain_pipe.predict(train_df['text'])
    acc_train = accuracy_score(train_df['domain'], preds_train)
    print(f"Domain predictor accuracy (train set): {acc_train:.4f}")
    print("Classification report (train set):")
    print(classification_report(train_df['domain'], preds_train))

    # Validation on val set
    val_path = 'data/processed/val.csv'
    if os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        preds_val = domain_pipe.predict(val_df['text'])
        acc_val = accuracy_score(val_df['domain'], preds_val)
        print(f"\nDomain predictor accuracy (val set): {acc_val:.4f}")
        print("Classification report (val set):")
        print(classification_report(val_df['domain'], preds_val))
    else:
        print(f"Validation file {val_path} not found. Skipping val set evaluation.")

    os.makedirs('./results', exist_ok=True)
    joblib.dump(domain_pipe, './results/domain_pipe.joblib')
    print('Domain predictor trained and saved to ./results/domain_pipe.joblib')
