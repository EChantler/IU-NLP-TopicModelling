# Export requirements
conda list -e > requirements.txt


# Plan
1. Pull different datasets
2. Combine the datasets with a domain feature and sample for equal representation
3. Map to common target topic set
4. Clean the text data
5. Split into train,validate and test sets (keep separate validate and test sets to test cross domain performance)
6. Train and optimize the models (single model vs domain -> Target models)
7. Evaluate model
8. Predict