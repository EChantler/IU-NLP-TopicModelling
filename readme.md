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
2. [] Promote the model to prod
3. [] Load the two models via validate, evaluate, predict and api