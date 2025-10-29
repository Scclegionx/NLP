from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import PhoBERTPredictor
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(
    title="PhoBERT NLP Server",
    description="NLP Server for Vietnamese Intent Classification and Entity Extraction",
    version="1.0.0"
)

# Global predictor instance
predictor = None

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    command: str
    entities: dict
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global predictor
    try:
        predictor = PhoBERTPredictor()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PhoBERT NLP Server is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_intent(input_data: TextInput):
    """
    Predict intent and extract entities from Vietnamese text
    
    Args:
        input_data: TextInput containing the text to analyze
        
    Returns:
        PredictionResponse with intent, entities, and other information
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        result = predictor.predict(input_data.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(input_data: list[TextInput]):
    """
    Predict intent for multiple texts at once
    
    Args:
        input_data: List of TextInput containing texts to analyze
        
    Returns:
        List of PredictionResponse
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = []
        for item in input_data:
            result = predictor.predict(item.text)
            results.append(PredictionResponse(**result))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
