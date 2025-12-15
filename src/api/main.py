from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import List
import uvicorn
from src.api.database import PredictionLogger

# Initialize FastAPI app
app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="API for predicting Remaining Useful Life of turbofan engines",
    version="1.0.0"
)

# Global model variable
model = None
feature_names = None

class PredictionRequest(BaseModel):
    """Request schema for prediction."""
    features: dict
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "sensor_2": 642.5,
                    "sensor_3": 1589.2,
                    "sensor_4": 1400.6,
                    "sensor_7": 554.8,
                    "sensor_8": 2388.1
                }
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    predicted_rul: float
    model_version: str
    confidence: str

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, feature_names
    
    try:
        # Load from local path (not MLflow)
        model = mlflow.xgboost.load_model("/app/model")
        
        train_data = pd.read_csv('/app/data/processed/train_FD001_processed.csv')
        feature_names = [col for col in train_data.columns if col not in ['unit', 'cycle', 'RUL']]
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Features: {len(feature_names)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Turbofan RUL Prediction API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "turbofan_rul_xgboost",
        "model_stage": "Staging",
        "n_features": len(feature_names),
        "feature_names": feature_names[:10]  # First 10 features
    }

prediction_logger = PredictionLogger()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make RUL prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create feature vector with zeros
        feature_vector = pd.DataFrame([{name: 0.0 for name in feature_names}])
        
        # Update with provided features
        for key, value in request.features.items():
            if key in feature_names:
                feature_vector[key] = value
        
        # Make prediction
        prediction = model.predict(feature_vector)[0]
        
        # Determine confidence based on prediction range
        if prediction < 0:
            prediction = 0
            confidence = "low"
        elif prediction > 200:
            confidence = "low"
        else:
            confidence = "high"

        prediction_logger.log_prediction(
        features=request.features,
        predicted_rul=float(prediction),
        model_version="Staging"
    )
        
        return PredictionResponse(
            predicted_rul=float(prediction),
            model_version="Staging",
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Make batch predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for req in requests:
            # Create feature vector
            feature_vector = pd.DataFrame([{name: 0.0 for name in feature_names}])
            
            # Update with provided features
            for key, value in req.features.items():
                if key in feature_names:
                    feature_vector[key] = value
            
            # Make prediction
            pred = model.predict(feature_vector)[0]
            
            if pred < 0:
                pred = 0
                confidence = "low"
            elif pred > 200:
                confidence = "low"
            else:
                confidence = "high"
            
            predictions.append({
                "predicted_rul": float(pred),
                "model_version": "Staging",
                "confidence": confidence
            })
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)