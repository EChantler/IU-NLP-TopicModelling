# #%% Import necessary libraries
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV
# import os
# import torch
# import multiprocessing as mp

# # 1) Force all libraries into single-threaded/serialized operation
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"]       = "1"
# os.environ["MKL_NUM_THREADS"]       = "1"
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

# # 2) Change the multiprocessing start method (macOS default is 'spawn')
# mp.set_start_method("forkserver", force=True)

# #%% Load the preprocessed data
# train_df = pd.read_csv('./data/processed/train.csv')
# val_df = pd.read_csv('./data/processed/val.csv')
# test_df = pd.read_csv('./data/processed/test.csv')
# # Split the target by "." and take the first part
# # train_df['target'] = train_df['target'].apply(lambda x: x.split('.')[0])
# # val_df['target'] = val_df['target'].apply(lambda x: x.split('.')[0])
# # test_df['target'] = test_df['target'].apply(lambda x: x.split('.')[0])
# # label_names = sorted(train_df['target'].unique())  # e.g. ['alt','comp',…,'talk']
# # label2id = {l:i for i,l in enumerate(label_names)}
# # label_names = train_df['target'].drop_duplicates().tolist()  # e.g. ['alt.atheism', 'comp.graphics', …, 'talk.politics.misc']
# # train_df['label_id'] = train_df['target'].map(label2id).astype(int)
# # val_df  ['label_id'] = val_df  ['target'].map(label2id).astype(int)
# # #%%
# from datasets import Dataset
# from transformers import AutoTokenizer

# # assume train_df has columns "clean_text" and "target"
# ds_train = Dataset.from_pandas(train_df.rename(columns={'text':'text','labels':'labels'}))
# ds_val   = Dataset.from_pandas(val_df.rename(columns={'text':'text','labels':'labels'}))
# #%%
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# def tokenize_fn(batch):
#     # **both** truncation and padding are required for batched tensors
#     return tokenizer(
#         batch["text"],
#         padding="max_length",
#         truncation=True,
#         max_length=128
#     )

# ds_train = ds_train.map(tokenize_fn, batched=True)
# ds_val   = ds_val.map(tokenize_fn, batched=True)

# # drop unused columns, keep only input_ids, attention_mask, and label
# ds_train = ds_train.remove_columns([col for col in ["text", "__index_level_0__"] if col in ds_train.column_names])
# ds_val = ds_val.remove_columns([col for col in ["text", "__index_level_0__"] if col in ds_val.column_names])
# # ds_train = ds_train.rename_column("target", "labels")
# # ds_val   = ds_val.rename_column("target", "labels")
# ds_train.set_format("torch")
# ds_val.set_format("torch")
# # print(ds_train.features)
# # print(ds_train[0])

# # %%
# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased",
#     num_labels = len(train_df['labels'].unique()),
# )

# #%%
# from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
# import evaluate

# # pick a directory for checkpoints
# args = TrainingArguments(
#   output_dir    = "./tfm-checkpoints",
#   eval_strategy = "epoch",
#   save_strategy       = "epoch",
#   learning_rate  = 2e-5,
#   per_device_train_batch_size = 16,
#   per_device_eval_batch_size  = 32,
#   num_train_epochs = 3,
#   weight_decay     = 0.01,
#   logging_steps    = 50,
#   load_best_model_at_end = True,
#   metric_for_best_model  = "accuracy",
#   # ↓ critical for avoiding forks/threads issues
#     dataloader_num_workers=0,
#  use_cpu=True,
# )

# # define accuracy metric
# accuracy = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = logits.argmax(-1)
#     return accuracy.compute(predictions=preds, references=labels)

# data_collator = DataCollatorWithPadding(tokenizer)
# trainer = Trainer(
#   model            = model,
#   args             = args,
#   train_dataset    = ds_train,
#   eval_dataset     = ds_val,
#   data_collator        = data_collator,
#   compute_metrics  = compute_metrics,
# )

# # %%
# trainer.train()
# trainer.evaluate()

# # %%

# # %%
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch
import pandas as pd

# 1) Load your CSV with the **raw text** and integer labels
train_df = pd.read_csv("data/processed/train.csv")  # must contain columns "text" (str) and "target" (int)
val_df   = pd.read_csv("data/processed/val.csv")

# Force the right dtypes just in case:
train_texts  = train_df["text"].astype(str).tolist()
train_labels = train_df["target"].astype(int).tolist()

val_texts    = val_df["text"].astype(str).tolist()
val_labels   = val_df["target"].astype(int).tolist()

# 2) Build Datasets from pure Python dicts
ds_train = Dataset.from_dict({
    "text":   train_texts,
    "labels": train_labels,
})
ds_val = Dataset.from_dict({
    "text":   val_texts,
    "labels": val_labels,
})

# 3) (Optional) turn your "labels" into a ClassLabel feature for better reporting
label_feature = ClassLabel(num_classes=len(set(train_labels)))
ds_train = ds_train.cast_column("labels", label_feature)
ds_val   = ds_val.cast_column("labels", label_feature)

# 4) Tokenize – now we know `batch["text"]` is a list of Python strings
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

ds_train = ds_train.map(tokenize_batch, batched=True, remove_columns=["text"])
ds_val   = ds_val  .map(tokenize_batch, batched=True, remove_columns=["text"])

# 5) Set to PyTorch tensors
ds_train.set_format("torch", columns=["input_ids","attention_mask","labels"])
ds_val  .set_format("torch", columns=["input_ids","attention_mask","labels"])

# 6) Load model & Trainer as before
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels = ds_train.features["labels"].num_classes
)

data_collator = DataCollatorWithPadding(tokenizer)
training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    dataloader_num_workers=0,
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=data_collator,
    compute_metrics=lambda p: {
        "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
    }
)

trainer.train()

