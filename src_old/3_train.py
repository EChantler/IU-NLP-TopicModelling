#%% Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

#%% Load the preprocessed data
train_df = pd.read_csv('../data/processed/train.csv')
val_df = pd.read_csv('../data/processed/val.csv')
test_df = pd.read_csv('../data/processed/test.csv')
# Split the target by "." and take the first part
train_df['target'] = train_df['target'].apply(lambda x: x.split('.')[0])
val_df['target'] = val_df['target'].apply(lambda x: x.split('.')[0])
test_df['target'] = test_df['target'].apply(lambda x: x.split('.')[0])

#%% Train a model with TF-IDF and Naive Bayes
# Vectorize the text data 
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer(
    sublinear_tf=True,      # dampens very high term counts
    min_df=5,               # ignore very rare words
    max_df=0.5,             # ignore extremely common words
    ngram_range=(1,1),      # unigrams + bigrams
    stop_words='english' 
)
# Create a Multinomial Naive Bayes model
model = MultinomialNB()
# model = LinearSVC(C=1.0)

# Create a pipeline that combines the vectorizer and the model
model = make_pipeline(vectorizer, model)

# Train the model on the training data
model.fit(train_df['text'], train_df['target'])

#%% Evaluate the model on the validation data
from sklearn.metrics import accuracy_score, classification_report
val_predictions = model.predict(val_df['text'])
val_accuracy = accuracy_score(val_df['target'], val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(classification_report(val_df['target'], val_predictions))


# %% Grid search for hyperparameter tuning for Naive Bayes with TF-IDF

pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8),
    MultinomialNB()
)

param_grid = {
    "tfidfvectorizer__ngram_range": [(1,1), (1,2), (1,3)],
    "tfidfvectorizer__min_df": [1,5,10],
    "tfidfvectorizer__max_df": [.5, 0.8, 0.9],
    "tfidfvectorizer__sublinear_tf": [False, True],
    "multinomialnb__alpha": [0.1, 0.5, 1.0],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1_macro",   # or "accuracy"
    n_jobs=-1
)

# split the target by "." and take the first part
train_df['parent_target'] = train_df['target'].apply(lambda x: x.split('.')[0])

grid.fit(train_df['text'], train_df['parent_target'])
print("Best params:", grid.best_params_)
best_model = grid.best_estimator_
# %%
print("Best score:", grid.best_score_)

# %% Train a model with GridSearchCV for SVM with TF-IDF
# 1. Build the pipeline
pipeline = make_pipeline(
    TfidfVectorizer(),   # we'll tune its params in the grid
    LinearSVC()
)

# 2. Define your hyperparameter grid
param_grid = {
    # TF–IDF options
    "tfidfvectorizer__ngram_range": [(1,1), (1,2), (1,3), (2,2), (2,3)],
    "tfidfvectorizer__min_df": [1, 5, 10],
    "tfidfvectorizer__max_df": [0.5],#, 0.8, 0.9],
    #"tfidfvectorizer__sublinear_tf": [False, True],

    # SVM options
    "linearsvc__C": [0.1, 1, 10],
    #"linearsvc__max_iter": [1000, 2000],    # you can tune or leave it at default
    # "linearsvc__dual": [True, False],     # sometimes useful to try
}

# 3. Parent target extraction
train_df['parent_target'] = train_df['target'].apply(lambda x: x.split('.')[0])

# 4. Set up GridSearchCV
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1_macro",   # or "accuracy"
    n_jobs=-1,
    verbose=3
)

# 5. Fit on your chosen target column
grid.fit(
    train_df['text'],           # or 'clean_text' if you preprocessed
    train_df['parent_target']   # or 'target' if you’re still doing 20-way
)
#%% 
print("Best params:", grid.best_params_)
best_model = grid.best_estimator_
print("Best score:", grid.best_score_)
# %%
pipeline = make_pipeline(
    TfidfVectorizer(
        sublinear_tf=True,            # dampens very frequent terms
        min_df=2,                     # ignore tokens in fewer than 2 docs
        max_df=0.7,                   # ignore tokens in >70% of docs
        ngram_range=(1,2),            # unigrams + bigrams
        analyzer='word',
        stop_words='english'  
        ),       
    LogisticRegression(
        C=1.0,                        # inverse regularization strength
        solver='saga',                # handles large sparse inputs
        max_iter=5000,
        n_jobs=-1,
        random_state=42
    )
)

# %% train
# train_df['parent_target'] = train_df['target'].apply(lambda x: x.split('.')[0])

# switch to your chosen column—'clean_text' if you preprocessed, otherwise 'text'
pipeline.fit(train_df['text'], train_df['target'])


# %% evaluate on validation
val_preds = pipeline.predict(val_df['text'])
print("Validation Accuracy:", accuracy_score(val_df['target'], val_preds))
print(classification_report(val_df['target'], val_preds))
# %%
