# Export requirements

## Export full environment (recommended)
conda env export > environment.yml

## Create environment from environment.yml
conda env create -f environment.yml

## Update existing environment from environment.yml
conda env update -f environment.yml --prune


# Plan
1. [x] Pull different datasets 
2. [x] Combine the datasets with a domain feature and sample for equal representation
3. [x] Map to common target topic set
4. [x] Clean the text data
5. [x] Split into train,validate and test sets (keep separate validate and test sets to test cross domain performance)
6. [x] Train and optimize the models (single model vs domain -> Target models, transformer)
7. [x] Evaluate model
8. [x] Predict

1. [x] Retrain the Transformer to create the model
2. [x] Promote the model to prod
3. [x] Load the two models via validate, evaluate, predict and api

Rethink the report completely. I found that transformers can do better on raw data. So I will redo the cleaning for a baseline classical model, train a raw data transformer, train a cleaned data transformer to justify the raw, tune the transformer and demonstrate size scaling

1. [x] Redo topic mappings
2. [x] Split regex cleaning and lemmatization and stopword removal
3. [x] Tune the classic model - smaller dataset
4. [x] Train baseline TF-IDF + LR model
4. [x] Tune the transformer (Learning rate, epochs, warmup ratio) - smaller dataset
4. [x] Train Transformer with regex cleaned data (10% data)
5. [x] Train transformer on lemmatized and stopword removed data (10% data)
6. [x] Train classic model and transformer on 25%, 50%, 75% and 100% datasets
7. [x] Evaluate on full datasets
8. [x] Do data size scaling f1&accuracy plot - with smaller dataset optimal parameters
* the idea is to show that the transformer outperforms the classic model and that lemmatization and stopword removal harms the transformer performance and that more data leads to better performance on the transformer.