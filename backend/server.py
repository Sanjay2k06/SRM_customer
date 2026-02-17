from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import joblib
import json
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Model paths
MODEL_DIR = ROOT_DIR / 'model'

# Global model variables
models_loaded = False
intent_model = None
loyalty_model = None
amount_model = None
scaler = None
label_encoders = None
intent_encoder = None
metadata = None
eda_data = None

def load_models():
    """Load all trained models and encoders"""
    global models_loaded, intent_model, loyalty_model, amount_model
    global scaler, label_encoders, intent_encoder, metadata, eda_data
    
    try:
        if not MODEL_DIR.exists():
            logging.warning("Model directory not found. Training models...")
            import subprocess
            subprocess.run(['python3', str(ROOT_DIR / 'train.py')], check=True)
        
        intent_model = joblib.load(MODEL_DIR / 'purchase_intent_model.pkl')
        loyalty_model = joblib.load(MODEL_DIR / 'loyalty_model.pkl')
        amount_model = joblib.load(MODEL_DIR / 'amount_model.pkl')
        scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
        label_encoders = joblib.load(MODEL_DIR / 'label_encoders.pkl')
        intent_encoder = joblib.load(MODEL_DIR / 'intent_encoder.pkl')
        
        with open(MODEL_DIR / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        with open(MODEL_DIR / 'eda_data.json', 'r') as f:
            eda_data = json.load(f)
        
        models_loaded = True
        logging.info("✓ All models loaded successfully")
        
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise

# Create the main app
app = FastAPI(title="Customer Intelligence API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Pydantic Models
class PredictionInput(BaseModel):
    age: int
    gender: str
    income_level: str
    marital_status: str
    education_level: str
    occupation: str
    purchase_category: str
    frequency_of_purchase: int
    purchase_channel: str
    brand_loyalty: int
    product_rating: int
    time_spent_on_research: float
    social_media_influence: str
    discount_sensitivity: str
    return_rate: int
    customer_satisfaction: int
    engagement_with_ads: str
    device_used: str
    payment_method: str
    time_to_decision: int

class PurchaseIntentResponse(BaseModel):
    prediction: str
    probability: Dict[str, float]
    confidence: float

class LoyaltyResponse(BaseModel):
    prediction: bool
    probability: float
    confidence: float

class AmountResponse(BaseModel):
    predicted_amount: float
    confidence_interval: Dict[str, float]

class PredictionHistory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prediction_type: str
    input_data: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

def encode_input(input_data: PredictionInput) -> np.ndarray:
    """Encode input data using saved encoders and scaler"""
    feature_dict = {
        'Age': input_data.age,
        'Gender': input_data.gender,
        'Income_Level': input_data.income_level,
        'Marital_Status': input_data.marital_status,
        'Education_Level': input_data.education_level,
        'Occupation': input_data.occupation,
        'Purchase_Category': input_data.purchase_category,
        'Frequency_of_Purchase': input_data.frequency_of_purchase,
        'Purchase_Channel': input_data.purchase_channel,
        'Brand_Loyalty': input_data.brand_loyalty,
        'Product_Rating': input_data.product_rating,
        'Time_Spent_on_Product_Research(hours)': input_data.time_spent_on_research,
        'Social_Media_Influence': input_data.social_media_influence,
        'Discount_Sensitivity': input_data.discount_sensitivity,
        'Return_Rate': input_data.return_rate,
        'Customer_Satisfaction': input_data.customer_satisfaction,
        'Engagement_with_Ads': input_data.engagement_with_ads,
        'Device_Used_for_Shopping': input_data.device_used,
        'Payment_Method': input_data.payment_method,
        'Time_to_Decision': input_data.time_to_decision
    }
    
    # Encode categorical features
    for col, encoder in label_encoders.items():
        if col in feature_dict:
            try:
                feature_dict[col] = encoder.transform([str(feature_dict[col])])[0]
            except ValueError:
                # Handle unseen categories
                feature_dict[col] = encoder.transform([encoder.classes_[0]])[0]
    
    # Create feature array in correct order
    feature_array = np.array([feature_dict[col] for col in metadata['feature_columns']]).reshape(1, -1)
    
    # Scale features
    scaled_features = scaler.transform(feature_array)
    
    return scaled_features

# API Endpoints
@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@api_router.post("/predict/purchase-intent", response_model=PurchaseIntentResponse)
async def predict_purchase_intent(input_data: PredictionInput):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Encode and predict
        X = encode_input(input_data)
        prediction = intent_model.predict(X)[0]
        probabilities = intent_model.predict_proba(X)[0]
        
        # Decode prediction
        predicted_class = intent_encoder.inverse_transform([prediction])[0]
        
        # Create probability dict
        prob_dict = {intent_encoder.classes_[i]: float(probabilities[i]) for i in range(len(probabilities))}
        confidence = float(max(probabilities))
        
        result = {
            "prediction": predicted_class,
            "probability": prob_dict,
            "confidence": confidence
        }
        
        # Save to history
        history = PredictionHistory(
            prediction_type="purchase_intent",
            input_data=input_data.model_dump(),
            result=result
        )
        doc = history.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.predictions.insert_one(doc)
        
        return result
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/predict/loyalty", response_model=LoyaltyResponse)
async def predict_loyalty(input_data: PredictionInput):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        X = encode_input(input_data)
        prediction = loyalty_model.predict(X)[0]
        probabilities = loyalty_model.predict_proba(X)[0]
        
        result = {
            "prediction": bool(prediction),
            "probability": float(probabilities[1]),
            "confidence": float(max(probabilities))
        }
        
        # Save to history
        history = PredictionHistory(
            prediction_type="loyalty",
            input_data=input_data.model_dump(),
            result=result
        )
        doc = history.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.predictions.insert_one(doc)
        
        return result
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/predict/amount", response_model=AmountResponse)
async def predict_amount(input_data: PredictionInput):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        X = encode_input(input_data)
        prediction = amount_model.predict(X)[0]
        
        # Calculate confidence interval (rough estimate)
        std_dev = prediction * 0.15  # 15% std deviation
        
        result = {
            "predicted_amount": float(prediction),
            "confidence_interval": {
                "lower": float(prediction - std_dev),
                "upper": float(prediction + std_dev)
            }
        }
        
        # Save to history
        history = PredictionHistory(
            prediction_type="amount",
            input_data=input_data.model_dump(),
            result=result
        )
        doc = history.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.predictions.insert_one(doc)
        
        return result
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/model-metrics")
async def get_model_metrics():
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return metadata

@api_router.get("/eda-data")
async def get_eda_data():
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return eda_data

@api_router.get("/dataset-info")
async def get_dataset_info():
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Load and preview dataset
    import pandas as pd
    df = pd.read_csv(ROOT_DIR.parent / 'data' / 'dataset.csv')
    
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "sample_data": df.head(5).to_dict(orient='records'),
        "statistics": df.describe().to_dict()
    }

@api_router.get("/prediction-history")
async def get_prediction_history(limit: int = 20):
    predictions = await db.predictions.find({}, {"_id": 0}).sort("timestamp", -1).to_list(limit)
    return predictions

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up...")
    load_models()
    logger.info("Application ready!")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
