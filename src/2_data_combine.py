import os
import pandas as pd

# Define the mapping from filename to domain
file_domain_map = {
    'news.csv': 'news',
    'social.csv': 'social',
    'academic.csv': 'academic',
}

raw_dir = os.path.join('data', 'raw')
all_dfs = []

# Find all relevant files in the raw data directory
for fname, domain in file_domain_map.items():
    fpath = os.path.join(raw_dir, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df['domain'] = domain
        all_dfs.append(df)

# If no files found, exit
if not all_dfs:
    raise RuntimeError('No domain CSV files found in data/raw.')

# Find the minimum sample size for equal representation
min_size = min(len(df) for df in all_dfs)

# For each domain, take the min_size or the full set if smaller
sampled_dfs = [
    df.sample(n=min_size, random_state=42) if len(df) >= min_size else df.copy()
    for df in all_dfs
]

# Concatenate and shuffle
combined_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to combined.csv in data/raw
combined_path = os.path.join(raw_dir, 'combined.csv')
combined_df.to_csv(combined_path, index=False)
print(f'Combined dataset saved to {combined_path}')
