from fastapi import FastAPI, Query
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
import uvicorn
import joblib
import os
from text_cleaning import clean_text


# Load domain predictor
domain_pipe_path = './results/domain_pipe.joblib'
if not os.path.exists(domain_pipe_path):
    raise FileNotFoundError(f"Domain predictor not found at {domain_pipe_path}. Please run train_domain_predictor.py first.")
domain_pipe = joblib.load(domain_pipe_path)

# Load model and tokenizer at startup
model_path = './results/final_transformer_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load label mapping from train.csv
train_df = pd.read_csv('data/processed/train.csv')
labels = sorted(train_df['target'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

app = FastAPI(title="Topic Prediction API", description="Predicts topic for a given text and domain.")


class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    topic: str


@app.post("/predict", response_model=PredictResponse)
def predict_topic(request: PredictRequest):
    # Clean text for domain prediction (must match training)
    cleaned_text = clean_text(request.text)
    pred_domain = domain_pipe.predict([cleaned_text])[0]
    input_text = f"{pred_domain} [SEP] {request.text}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        topic = id2label[pred_id]
    return PredictResponse(topic=topic)

# For local dev: run with `python src/api.py`
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
