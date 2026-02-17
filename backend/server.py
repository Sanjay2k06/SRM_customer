from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
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
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
try:
    client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=2000)
    db = client[os.environ.get('DB_NAME', 'test_database')]
    mongodb_available = True
except Exception as e:
    logging.warning(f"MongoDB not available: {str(e)}. Running in memory mode.")
    db = None
    client = None
    mongodb_available = False

# Model paths
MODEL_DIR = ROOT_DIR / 'model'
DATASET_PATH = ROOT_DIR.parent / 'Ecommerce_Consumer_Behavior_Analysis_Data.csv'

# Global model variables
models_loaded = False
intent_model = None
loyalty_model = None
amount_model = None
satisfaction_model = None
return_rate_model = None
scaler = None
label_encoders = None
intent_encoder = None
metadata = None
eda_data = None
research_time_max = None

def load_models():
    """Load all trained models and encoders"""
    global models_loaded, intent_model, loyalty_model, amount_model
    global scaler, label_encoders, intent_encoder, metadata, eda_data
    global satisfaction_model, return_rate_model
    global research_time_max
    
    try:
        if not MODEL_DIR.exists():
            logging.warning("Model directory not found. Training models...")
            import subprocess
            subprocess.run(['python', str(ROOT_DIR / 'train.py')], check=True)
        
        intent_model = joblib.load(MODEL_DIR / 'purchase_intent_model.pkl')
        loyalty_model = joblib.load(MODEL_DIR / 'loyalty_model.pkl')
        amount_model = joblib.load(MODEL_DIR / 'amount_model.pkl')
        satisfaction_model = joblib.load(MODEL_DIR / 'satisfaction_model.pkl')
        return_rate_model = joblib.load(MODEL_DIR / 'return_rate_model.pkl')
        scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
        label_encoders = joblib.load(MODEL_DIR / 'label_encoders.pkl')
        intent_encoder = joblib.load(MODEL_DIR / 'intent_encoder.pkl')
        
        with open(MODEL_DIR / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        with open(MODEL_DIR / 'eda_data.json', 'r') as f:
            eda_data = json.load(f)

        # Cache dataset stats used in feature engineering
        try:
            if DATASET_PATH.exists():
                df_stats = pd.read_csv(DATASET_PATH)
                research_time_max = float(df_stats['Time_Spent_on_Product_Research(hours)'].max())
            else:
                research_time_max = 0.0
        except Exception:
            research_time_max = 0.0
        
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

class SatisfactionResponse(BaseModel):
    predicted_satisfaction: float
    confidence: float

class ReturnRateResponse(BaseModel):
    will_return: bool
    probability: float
    confidence: float

class AnalyticsResponse(BaseModel):
    feature_importance: Dict[str, float]
    top_features: List[str]
    model_performance: Dict[str, float]

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionInput]

class BatchPredictionResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_processed: int
    timestamp: str

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
    
    # Feature engineering to match training pipeline
    if 'age_income_interaction' in metadata['feature_columns']:
        income_encoded = feature_dict.get('Income_Level', 0)
        feature_dict['age_income_interaction'] = feature_dict['Age'] * (income_encoded + 1)
    if 'satisfaction_loyalty_interaction' in metadata['feature_columns']:
        feature_dict['satisfaction_loyalty_interaction'] = (
            feature_dict['Customer_Satisfaction'] * feature_dict['Brand_Loyalty']
        )
    if 'research_time_normalized' in metadata['feature_columns']:
        max_val = research_time_max or feature_dict['Time_Spent_on_Product_Research(hours)']
        feature_dict['research_time_normalized'] = (
            feature_dict['Time_Spent_on_Product_Research(hours)'] / (max_val + 1)
        )

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
        if mongodb_available and db:
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
        if mongodb_available and db:
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
        if mongodb_available and db:
            doc = history.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            await db.predictions.insert_one(doc)
        
        return result
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/predict/satisfaction", response_model=SatisfactionResponse)
async def predict_satisfaction(input_data: PredictionInput):
    """Predict customer satisfaction level"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        X = encode_input(input_data)
        prediction = satisfaction_model.predict(X)[0]
        
        # Get prediction bounds (1-10 scale for satisfaction)
        pred_clipped = np.clip(prediction, 1, 10)
        
        result = {
            "predicted_satisfaction": float(pred_clipped),
            "confidence": 0.85
        }
        
        history = PredictionHistory(
            prediction_type="satisfaction",
            input_data=input_data.model_dump(),
            result=result
        )
        if mongodb_available and db:
            doc = history.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            await db.predictions.insert_one(doc)
        
        return result
        
    except Exception as e:
        logging.error(f"Satisfaction prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/predict/return-rate", response_model=ReturnRateResponse)
async def predict_return_rate(input_data: PredictionInput):
    """Predict if customer will return the product"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        X = encode_input(input_data)
        prediction = return_rate_model.predict(X)[0]
        probabilities = return_rate_model.predict_proba(X)[0]
        
        result = {
            "will_return": bool(prediction),
            "probability": float(max(probabilities)),
            "confidence": float(max(probabilities))
        }
        
        history = PredictionHistory(
            prediction_type="return_rate",
            input_data=input_data.model_dump(),
            result=result
        )
        if mongodb_available and db:
            doc = history.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            await db.predictions.insert_one(doc)
        
        return result
        
    except Exception as e:
        logging.error(f"Return rate prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(batch_request: BatchPredictionRequest):
    """Process batch predictions for multiple customers"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        results = []
        for idx, input_data in enumerate(batch_request.predictions):
            try:
                X = encode_input(input_data)
                
                intent_pred = intent_model.predict(X)[0]
                intent_proba = intent_model.predict_proba(X)[0]
                intent_class = intent_encoder.inverse_transform([intent_pred])[0]
                
                amount_pred = amount_model.predict(X)[0]
                loyalty_pred = loyalty_model.predict(X)[0]
                loyalty_proba = loyalty_model.predict_proba(X)[0]
                satisfaction_pred = satisfaction_model.predict(X)[0]
                
                results.append({
                    "record_index": idx,
                    "purchase_intent": intent_class,
                    "intent_confidence": float(max(intent_proba)),
                    "predicted_amount": float(amount_pred),
                    "loyalty_prediction": bool(loyalty_pred),
                    "loyalty_confidence": float(max(loyalty_proba)),
                    "satisfaction_score": float(np.clip(satisfaction_pred, 1, 10))
                })
            except Exception as e:
                logging.error(f"Batch prediction error for record {idx}: {str(e)}")
                results.append({
                    "record_index": idx,
                    "error": str(e)
                })
        
        return {
            "results": results,
            "total_processed": len(batch_request.predictions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
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
    df = pd.read_csv(DATASET_PATH)
    
    # Clean Purchase_Amount for proper parsing
    if 'Purchase_Amount' in df.columns:
        df['Purchase_Amount'] = df['Purchase_Amount'].apply(
            lambda x: float(str(x).replace('$', '').replace(',', '').strip()) if isinstance(x, str) else x
        )
    
    # Get numeric columns only for statistics
    numeric_df = df.select_dtypes(include=[np.number])
    stats_dict = numeric_df.describe().to_dict()
    
    # Replace NaN, Inf with None for JSON serialization
    for col in stats_dict:
        for stat in stats_dict[col]:
            val = stats_dict[col][stat]
            if pd.isna(val) or np.isinf(val):
                stats_dict[col][stat] = None
    
    # Clean sample data
    sample_data = df.head(5).to_dict(orient='records')
    for record in sample_data:
        for key, val in record.items():
            if isinstance(val, (float, np.floating)) and (pd.isna(val) or np.isinf(val)):
                record[key] = None
    
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "sample_data": sample_data,
        "statistics": stats_dict
    }

@api_router.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...), retrain: bool = True):
    """Upload a CSV or XML dataset and optionally retrain models."""
    filename = (file.filename or "").lower()
    if not (filename.endswith(".csv") or filename.endswith(".xml")):
        raise HTTPException(status_code=400, detail="Only .csv and .xml files are supported")

    try:
        content = await file.read()
        if filename.endswith(".csv"):
            df = pd.read_csv(pd.io.common.BytesIO(content))
        else:
            df = pd.read_xml(pd.io.common.BytesIO(content))

        if df is None or df.empty:
            raise ValueError("Uploaded file contains no data")

        df.to_csv(DATASET_PATH, index=False)

        if retrain:
            import subprocess
            subprocess.run(['python', str(ROOT_DIR / 'train.py')], check=True)
            load_models()

        return {
            "message": "Dataset uploaded successfully",
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "retrained": retrain
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Dataset upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/prediction-history")
async def get_prediction_history(limit: int = 20):
    if not mongodb_available or not db:
        return {"message": "Database not available", "predictions": []}
    try:
        predictions = await db.predictions.find({}, {"_id": 0}).sort("timestamp", -1).to_list(limit)
        return predictions
    except Exception as e:
        logging.warning(f"Could not retrieve predictions: {str(e)}")
        return {"message": "Error retrieving predictions", "predictions": []}

@api_router.get("/analytics/overview")
async def get_analytics_overview():
    """Get comprehensive analytics overview from EDA data"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "dataset_overview": eda_data.get("dataset_overview", {}),
        "top_categories": eda_data.get("top_categories", {}),
        "loyalty_stats": eda_data.get("loyalty_stats", {}),
        "discount_impact": eda_data.get("discount_impact", {}),
        "channel_performance": eda_data.get("channel_performance", {}),
        "correlation_analysis": eda_data.get("correlation_analysis", {})
    }

@api_router.get("/analytics/demographics")
async def get_demographics():
    """Get demographic insights"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return eda_data.get("demographic_insights", {})

@api_router.get("/analytics/satisfaction")
async def get_satisfaction_analytics():
    """Get satisfaction metrics"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return eda_data.get("satisfaction_metrics", {})

@api_router.get("/analytics/device-performance")
async def get_device_performance():
    """Get device usage and performance metrics"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return eda_data.get("device_insights", {})

@api_router.get("/model-insights")
async def get_model_insights():
    """Get model training details and performance"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "models_info": metadata.get("models_used", {}),
        "performance_metrics": metadata.get("metrics", {}),
        "dataset_info": metadata.get("dataset_info", {}),
        "feature_count": len(metadata.get("feature_columns", []))
    }

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
    if client:
        client.close()
