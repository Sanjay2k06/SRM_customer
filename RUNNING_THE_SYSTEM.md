# Running the System

## Current Status

✅ External branding references removed
✅ Models trained successfully on real dataset
✅ Backend API fully implemented with 15+ endpoints
✅ Frontend dashboard enhanced with 5 prediction types
✅ Environment configured for localhost

## Starting the Application

### Step 1: Train Models (First Time Only)

```bash
cd backend
python train_final.py
```

This will:
- Load the real e-commerce dataset (1,000 records)
- Train 5 ML models using XGBoost, RandomForest, and GradientBoosting
- Generate EDA analytics data
- Save all models to `backend/model/` directory

Expected output:
```
============================================================
ADVANCED ML MODEL TRAINING PIPELINE
With Real E-commerce Dataset
============================================================

1. Loading dataset...
   Dataset shape: (1000, 28)
   Total samples: 1000

2. Cleaning data...
   [+] Missing values handled
   [+] Purchase amount cleaned

3. Advanced Feature Engineering...
   [+] Encoded 12 categorical features
   [+] Added interaction features

4. Training Purchase Intent Models (Ensemble)...
   [+] XGBoost Accuracy: 31.00%
   ...

[OK] ADVANCED MODEL TRAINING COMPLETED SUCCESSFULLY
```

### Step 2: Start Backend Server

```bash
cd backend
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
Using selector: WinSelectorEventLoop
Starting up...
✓ All models loaded successfully
Application ready!
```

### Step 3: Start Frontend Server (New Terminal)

```bash
cd frontend
npm install  # First time only
npm start
```

Expected output:
```
> react-scripts start

webpack compiled successfully
Compiled successfully!

You can now view the app in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000

Note that the development build is not optimized.
To create a production build, use npm run build.
```

## Accessing the Application

Once both servers are running:

1. **Open Dashboard**: http://localhost:3000
   - You should see the Customer Intelligence Dashboard
   - Dark theme with blue gradient background

2. **API Documentation**: http://localhost:8000/docs
   - Interactive Swagger UI
   - Try out API endpoints directly

3. **Health Check**: http://localhost:8000/api/health
   - Should return: `{"status": "healthy", "models_loaded": true, ...}`

## Using the Dashboard

### 1. View Exploratory Analysis
- Tab: **Exploratory Analysis**
- See income vs spending patterns
- Check purchase intent distribution
- Analyze loyalty program impact
- Review discount effectiveness

### 2. Make Predictions
- Tab: **Predictions**
- Fill in customer information (age, income, preferences, etc.)
- Click prediction buttons:
  - **Predict Purchase Intent** - Type of purchase (Impulsive, Need-based, etc.)
  - **Predict Loyalty** - Will they join loyalty program?
  - **Predict Spending** - How much will they spend?
  - **Predict Satisfaction** - Customer satisfaction score (1-10)
  - **Predict Return Risk** - Will they return the product?

### 3. Explore Dataset
- Tab: **Dataset Overview**
- See data shape and structure
- Check sample records
- View statistical summaries

### 4. Get Business Insights
- Tab: **Business Insights**
- Read AI-generated recommendations
- See revenue optimization tips
- Get customer segmentation insights
- Get actionable next steps

## Sample Predictions

### Good Input Example:
```json
{
  "age": 35,
  "gender": "Male",
  "income_level": "High",
  "education_level": "Master's",
  "occupation": "High",
  "purchase_category": "Electronics",
  "frequency_of_purchase": 8,
  "purchase_channel": "Online",
  "brand_loyalty": 4,
  "product_rating": 4,
  "time_spent_on_research": 2.5,
  "social_media_influence": "High",
  "discount_sensitivity": "Not Sensitive",
  "return_rate": 0,
  "customer_satisfaction": 8,
  "engagement_with_ads": "High",
  "device_used": "Smartphone",
  "payment_method": "Credit Card",
  "marital_status": "Married",
  "time_to_decision": 5
}
```

Expected results:
- Purchase Intent: 45% confidence
- Loyalty: 72% probability
- Spending: $245.50 ± $35

## API Endpoints Available

### Predictions
- `POST /api/predict/purchase-intent` - Classify purchase type
- `POST /api/predict/loyalty` - Predict loyalty membership
- `POST /api/predict/amount` - Forecast spending
- `POST /api/predict/satisfaction` - Score satisfaction (NEW)
- `POST /api/predict/return-rate` - Assess return risk (NEW)

### Analytics
- `GET /api/analytics/overview` - Dashboard metrics
- `GET /api/analytics/demographics` - Customer demographics
- `GET /api/analytics/satisfaction` - Satisfaction metrics
- `GET /api/analytics/device-performance` - Device insights
- `GET /api/model-insights` - Model performance data

### Monitoring
- `GET /api/health` - Service status
- `GET /api/model-metrics` - Model accuracy metrics
- `GET /api/eda-data` - Full analytics data
- `GET /api/dataset-info` - Dataset statistics
- `GET /api/prediction-history` - Past predictions (if DB available)

## Stopping the Servers

Press `Ctrl+C` in each terminal to stop the servers gracefully.

## Troubleshooting

### Port Already in Use
```bash
# Windows - Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8000
kill -9 <PID>
```

### Module Not Found
```bash
pip install -r backend/requirements.txt
```

### Dataset Not Found
- Ensure `Ecommerce_Consumer_Behavior_Analysis_Data.csv` is in the root directory
- Path should be: `SRM_customer/Ecommerce_Consumer_Behavior_Analysis_Data.csv`

### CORS Issues
- Backend configured to accept all origins
- Check `.env` files in both frontend and backend

### Frontend Not Connecting
- Verify backend is running on port 8000
- Check `frontend/.env` has correct URL
- Clear browser cache and refresh

## Performance Expectations

- Model training: 1-2 minutes
- Backend startup: 10-15 seconds
- First prediction: 2-3 seconds
- Subsequent predictions: <1 second
- Dashboard load: 5 seconds

## Next Development Steps

1. **Fine-tune Models**
   - Run hyperparameter optimization
   - Implement cross-validation
   - Test on larger datasets

2. **Add Features**
   - Batch prediction endpoint
   - Export predictions to CSV
   - Save customer profiles
   - Track prediction accuracy

3. **Database Integration**
   - Setup MongoDB for persistence
   - Store prediction history
   - Build customer database
   - Analytics dashboards

4. **Production Deployment**
   - Docker containerization
   - Cloud deployment (AWS/GCP)
   - CI/CD pipeline
   - Monitoring and logging

## Support

All documentation is in the root directory:
- `QUICKSTART.md` - Quick reference
- `IMPLEMENTATION_SUMMARY.md` - Full technical details
- `README.md` - Project overview

---

**System ready for development!** 🚀
