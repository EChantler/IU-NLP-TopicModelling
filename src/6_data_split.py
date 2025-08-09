import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

# MLflow run start
mlflow.start_run()

# Log data split parameters
mlflow.log_param("train_size", 0.7)
mlflow.log_param("temp_size", 0.3)
mlflow.log_param("val_size", 0.5)
mlflow.log_param("random_state", 42)

# Load the cleaned data
in_path = os.path.join('.', 'data', 'processed', 'combined_cleaned.csv')
df = pd.read_csv(in_path)

# Split into train (70%), temp (30%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['domain'])

# Split temp into val (15%) and test (15%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['domain'])

# Save train and val
out_dir = os.path.join('.', 'data', 'processed')
os.makedirs(out_dir, exist_ok=True)
train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(out_dir, 'val.csv'), index=False)

test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
print(f"Saved train set for with {len(train_df)} samples.")
print(f"Saved val set for with {len(val_df)} samples.")


# Save test set split by domain
for domain in test_df['domain'].unique():
    domain_df = test_df[test_df['domain'] == domain]
    domain_df.to_csv(os.path.join(out_dir, f'test_{domain}.csv'), index=False)
    print(f"Saved test set for domain '{domain}' with {len(domain_df)} samples.")

print("Train/val/test splits complete.")

# Log artifacts to MLflow
mlflow.log_artifacts(out_dir, artifact_path="data_splits")

# End MLflow run
mlflow.end_run()
