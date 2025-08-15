from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import os
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NLP Topic Classification API",
    description="API for classifying text into academic topics using Logistic Regression and Transformer models",
    version="1.0.0"
)

# Global variables to store loaded models
lr_model = None
transformer_model = None
transformer_tokenizer = None
model_labels = None

# Request/Response models
class TextInput(BaseModel):
    text: str = Field(..., description="Text to classify", min_length=1, max_length=10000)

class PredictionResponse(BaseModel):
    predicted_topic: str = Field(..., description="Predicted topic category")
    confidence: float = Field(..., description="Confidence score (0-1)")
    prediction_time_ms: float = Field(..., description="Time taken for prediction in milliseconds")
    model_used: str = Field(..., description="Model used for prediction")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of texts to classify", min_items=1, max_items=100)

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_time_ms: float = Field(..., description="Total time for batch prediction")
    batch_size: int = Field(..., description="Number of texts processed")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    api_version: str

# Model loading functions
def load_logistic_regression_model():
    """Load the trained logistic regression model"""
    global lr_model
    try:
        # Try loading from local file first
        if os.path.exists('./results/target_pipe.joblib'):
            lr_model = joblib.load('./results/target_pipe.joblib')
            logger.info("✓ Loaded logistic regression model from local file")
            return True
        else:
            logger.warning("✗ Local logistic regression model not found")
    except Exception as e:
        logger.error(f"✗ Failed to load local logistic regression model: {e}")
    
    # Try loading from MLflow
    try:
        client = MlflowClient()
        model_name = "logistic_regression_target_predictor"
        
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if latest_versions:
            model_version = latest_versions[0].version
            model_uri = f"models:/{model_name}/{model_version}"
            lr_model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✓ Loaded logistic regression model from MLflow (version {model_version})")
            return True
        else:
            logger.error(f"✗ No versions found for model {model_name}")
            return False
    except Exception as e:
        logger.error(f"✗ Failed to load logistic regression model from MLflow: {e}")
        return False

def load_transformer_model():
    """Load the trained transformer model"""
    global transformer_model, transformer_tokenizer, model_labels
    
    # Try loading from local path first
    model_path = './results/final_transformer_model'
    try:
        if os.path.exists(model_path):
            transformer_tokenizer = AutoTokenizer.from_pretrained(model_path)
            transformer_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            transformer_model.eval()  # Set to evaluation mode
            
            # Extract label mappings
            if hasattr(transformer_model, 'config') and hasattr(transformer_model.config, 'id2label'):
                model_labels = transformer_model.config.id2label
            
            logger.info(f"✓ Loaded transformer model from local path: {model_path}")
            return True
        else:
            logger.warning(f"✗ Local transformer model path not found: {model_path}")
    except Exception as e:
        logger.error(f"✗ Failed to load transformer model from local path: {e}")
    
    # Try loading from MLflow
    try:
        client = MlflowClient()
        model_name = "transformer_model"
        
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if latest_versions:
            model_version = latest_versions[0].version
            model_uri = f"models:/{model_name}/{model_version}"
            
            transformer_model = mlflow.pytorch.load_model(model_uri)
            transformer_model.eval()
            transformer_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Try to get label mappings
            if hasattr(transformer_model, 'config') and hasattr(transformer_model.config, 'id2label'):
                model_labels = transformer_model.config.id2label
            
            logger.info(f"✓ Loaded transformer model from MLflow (version {model_version})")
            return True
        else:
            logger.error(f"✗ No versions found for model {model_name}")
            return False
    except Exception as e:
        logger.error(f"✗ Failed to load transformer model from MLflow: {e}")
        return False

def predict_with_lr(text: str) -> tuple:
    """Make prediction using logistic regression model"""
    if lr_model is None:
        raise HTTPException(status_code=503, detail="Logistic regression model not loaded")
    
    start_time = time.time()
    
    try:
        # Make prediction
        prediction = lr_model.predict([text])[0]
        
        # Get prediction probabilities for confidence
        probabilities = lr_model.predict_proba([text])[0]
        confidence = float(max(probabilities))
        
        prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return prediction, confidence, prediction_time
    
    except Exception as e:
        logger.error(f"Error during LR prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def predict_with_transformer(text: str) -> tuple:
    """Make prediction using transformer model"""
    if transformer_model is None or transformer_tokenizer is None:
        raise HTTPException(status_code=503, detail="Transformer model not loaded")
    
    start_time = time.time()
    
    try:
        # Use raw text without domain prefix
        input_text = text
        
        # Tokenize
        inputs = transformer_tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = transformer_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(logits, dim=1).item()
            confidence = float(torch.max(probabilities).item())
        
        # Convert to label if mapping exists
        if model_labels and predicted_class_id in model_labels:
            prediction = model_labels[predicted_class_id]
        else:
            prediction = str(predicted_class_id)
        
        prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return prediction, confidence, prediction_time
    
    except Exception as e:
        logger.error(f"Error during transformer prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NLP Topic Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_lr": "/predict/logistic-regression",
            "predict_transformer": "/predict/transformer",
            "batch_predict_lr": "/predict/batch/logistic-regression",
            "batch_predict_transformer": "/predict/batch/transformer"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={
            "logistic_regression": lr_model is not None,
            "transformer": transformer_model is not None and transformer_tokenizer is not None
        },
        api_version="1.0.0"
    )

@app.post("/predict/logistic-regression", response_model=PredictionResponse)
async def predict_logistic_regression(input_data: TextInput):
    """Predict topic using Logistic Regression model"""
    prediction, confidence, prediction_time = predict_with_lr(input_data.text)
    
    return PredictionResponse(
        predicted_topic=prediction,
        confidence=confidence,
        prediction_time_ms=prediction_time,
        model_used="logistic_regression"
    )

@app.post("/predict/transformer", response_model=PredictionResponse)
async def predict_transformer_endpoint(input_data: TextInput):
    """Predict topic using Transformer model"""
    prediction, confidence, prediction_time = predict_with_transformer(input_data.text)
    
    return PredictionResponse(
        predicted_topic=prediction,
        confidence=confidence,
        prediction_time_ms=prediction_time,
        model_used="transformer"
    )

@app.post("/predict/batch/logistic-regression", response_model=BatchPredictionResponse)
async def batch_predict_logistic_regression(input_data: BatchTextInput):
    """Batch predict topics using Logistic Regression model"""
    start_time = time.time()
    predictions = []
    
    for text in input_data.texts:
        prediction, confidence, pred_time = predict_with_lr(text)
        predictions.append(PredictionResponse(
            predicted_topic=prediction,
            confidence=confidence,
            prediction_time_ms=pred_time,
            model_used="logistic_regression"
        ))
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_time_ms=total_time,
        batch_size=len(input_data.texts)
    )

@app.post("/predict/batch/transformer", response_model=BatchPredictionResponse)
async def batch_predict_transformer(input_data: BatchTextInput):
    """Batch predict topics using Transformer model"""
    start_time = time.time()
    predictions = []
    
    for text in input_data.texts:
        prediction, confidence, pred_time = predict_with_transformer(text)
        predictions.append(PredictionResponse(
            predicted_topic=prediction,
            confidence=confidence,
            prediction_time_ms=pred_time,
            model_used="transformer"
        ))
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_time_ms=total_time,
        batch_size=len(input_data.texts)
    )

# Startup event to load models
@app.on_event("startup")
async def load_models():
    """Load models when the API starts"""
    logger.info("Loading models...")
    
    # Load logistic regression model
    lr_success = load_logistic_regression_model()
    if not lr_success:
        logger.warning("Failed to load logistic regression model - LR endpoints will be unavailable")
    
    # Load transformer model
    transformer_success = load_transformer_model()
    if not transformer_success:
        logger.warning("Failed to load transformer model - Transformer endpoints will be unavailable")
    
    if lr_success or transformer_success:
        logger.info("API startup completed - at least one model loaded successfully")
    else:
        logger.error("API startup completed - NO MODELS LOADED - all prediction endpoints will fail")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": [
        "/", "/health", "/predict/logistic-regression", "/predict/transformer",
        "/predict/batch/logistic-regression", "/predict/batch/transformer"
    ]}

if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    print("Starting NLP Topic Classification API...")
    print("Available endpoints:")
    print("  - GET  /                              : API information")
    print("  - GET  /health                        : Health check")
    print("  - POST /predict/logistic-regression   : Single prediction with LR model")
    print("  - POST /predict/transformer           : Single prediction with Transformer model")
    print("  - POST /predict/batch/logistic-regression : Batch prediction with LR model")
    print("  - POST /predict/batch/transformer     : Batch prediction with Transformer model")
    print("\nStarting server on http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(app, host="localhost", port=8000)
