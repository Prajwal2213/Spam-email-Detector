from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Email Spam Detector API")

# Load the trained model pipeline
MODEL_PATH = "model_pipeline.joblib"
if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
else:
    model_pipeline = None
    print("Warning: Model pipeline not found. Run train_model.py first.")

class PredictRequest(BaseModel):
    message: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    if not request.message or request.message.strip() == "":
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    # Predict
    # model_pipeline expects a list/array of strings
    prediction_num = model_pipeline.predict([request.message])[0]
    
    # Predict probabilities to get confidence
    probabilities = model_pipeline.predict_proba([request.message])[0]
    confidence = max(probabilities)
    
    label = "spam" if prediction_num == 1 else "ham"
    
    # User requested override: if confidence < 70%, mark it as spam
    if confidence < 0.70:
        label = "spam"
    
    return PredictResponse(prediction=label, confidence=confidence)

# Mount the frontend 'public' directory
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
if not os.path.exists(PUBLIC_DIR):
    os.makedirs(PUBLIC_DIR)
    
app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Please create public/index.html"}
