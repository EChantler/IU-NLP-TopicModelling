#%% Load the data
from sklearn.datasets import fetch_20newsgroups
cats = None #['alt.atheism', 'sci.space']
newsgroups = fetch_20newsgroups(data_home='../data', subset="all", categories=cats, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
#%% Split the data into train, validate and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data[:1000], newsgroups.target[:1000], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
#%% Print the shapes of the splits
print(f"Train shape: {len(X_train)}")
print(f"Validation shape: {len(X_val)}")
print(f"Test shape: {len(X_test)}")
#%% Save the data to csv files
import pandas as pd
train_df = pd.DataFrame({'text': X_train, 'target': y_train})
train_df.to_csv('../data/raw/train.csv', index=False)
val_df = pd.DataFrame({'text': X_val, 'target': y_val})
val_df.to_csv('../data/raw/val.csv', index=False)
test_df = pd.DataFrame({'text': X_test, 'target': y_test})
test_df.to_csv('../data/raw/test.csv', index=False)

#%% load the data from csv files
train_df = pd.read_csv('../data/raw/train.csv')
val_df = pd.read_csv('../data/raw/val.csv')
test_df = pd.read_csv('../data/raw/test.csv')
#%% Print the shape of the data
print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")
#%% Print the first 5 rows of the train data
print(train_df.head())
#%% Print the first 5 rows of the validation data
print(val_df.head())
#%% Print the first 5 rows of the test data
print(test_df.head())  