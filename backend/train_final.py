import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                             GradientBoostingClassifier, GradientBoostingRegressor)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, mean_squared_error, r2_score, mean_absolute_error)
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent
MODEL_DIR = ROOT_DIR / 'model'
DATA_PATH = ROOT_DIR.parent / 'Ecommerce_Consumer_Behavior_Analysis_Data.csv'

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)

def clean_purchase_amount(amount_str):
    """Clean purchase amount string to float"""
    if isinstance(amount_str, str):
        return float(amount_str.replace('$', '').replace(',', '').strip())
    return float(amount_str)

def train_models():
    print("=" * 60)
    print("ADVANCED ML MODEL TRAINING PIPELINE")
    print("With Real E-commerce Dataset")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Total samples: {len(df)}")
    
    # Data Cleaning
    print("\n2. Cleaning data...")
    df['Purchase_Amount'] = df['Purchase_Amount'].apply(clean_purchase_amount)
    df['Social_Media_Influence'].fillna('None', inplace=True)
    df['Engagement_with_Ads'].fillna('None', inplace=True)
    
    print("   [+] Missing values handled")
    print("   [+] Purchase amount cleaned")
    
    # Feature Engineering
    print("\n3. Advanced Feature Engineering...")
    
    feature_columns = [
        'Age', 'Gender', 'Income_Level', 'Marital_Status', 'Education_Level',
        'Occupation', 'Purchase_Category', 'Frequency_of_Purchase',
        'Purchase_Channel', 'Brand_Loyalty', 'Product_Rating',
        'Time_Spent_on_Product_Research(hours)', 'Social_Media_Influence',
        'Discount_Sensitivity', 'Return_Rate', 'Customer_Satisfaction',
        'Engagement_with_Ads', 'Device_Used_for_Shopping', 'Payment_Method',
        'Time_to_Decision'
    ]
    
    numerical_features = [
        'Age', 'Frequency_of_Purchase', 'Brand_Loyalty', 'Product_Rating',
        'Time_Spent_on_Product_Research(hours)', 'Return_Rate',
        'Customer_Satisfaction', 'Time_to_Decision'
    ]
    
    categorical_features = [
        'Gender', 'Income_Level', 'Marital_Status', 'Education_Level',
        'Occupation', 'Purchase_Category', 'Purchase_Channel',
        'Social_Media_Influence', 'Discount_Sensitivity',
        'Engagement_with_Ads', 'Device_Used_for_Shopping', 'Payment_Method'
    ]
    
    # Encode categorical variables
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    print(f"   [+] Encoded {len(categorical_features)} categorical features")
    
    # Create interaction features
    df_encoded['age_income_interaction'] = df_encoded['Age'] * (df_encoded['Income_Level'] + 1)
    df_encoded['satisfaction_loyalty_interaction'] = df_encoded['Customer_Satisfaction'] * df_encoded['Brand_Loyalty']
    df_encoded['research_time_normalized'] = df_encoded['Time_Spent_on_Product_Research(hours)'] / (df_encoded['Time_Spent_on_Product_Research(hours)'].max() + 1)
    
    feature_columns.extend(['age_income_interaction', 'satisfaction_loyalty_interaction', 'research_time_normalized'])
    print("   [+] Added interaction features")
    
    # Save encoders
    joblib.dump(label_encoders, MODEL_DIR / 'label_encoders.pkl')
    print("   [+] Label encoders saved")
    
    # Prepare features
    X = df_encoded[feature_columns]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
    print("   [+] Feature scaler saved")
    
    # ========================================
    # MODEL 1: Purchase Intent Classification
    # ========================================
    print("\n4. Training Purchase Intent Models (Ensemble)...")
    
    y_intent = df['Purchase_Intent']
    intent_encoder = LabelEncoder()
    y_intent_encoded = intent_encoder.fit_transform(y_intent)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_intent_encoded, test_size=0.2, random_state=42, stratify=y_intent_encoded
    )
    
    # XGBoost
    intent_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softprob',
        num_class=len(intent_encoder.classes_),
        eval_metric='mlogloss',
        verbosity=0
    )
    intent_xgb.fit(X_train, y_train, verbose=False)
    
    y_pred_xgb = intent_xgb.predict(X_test)
    intent_accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    
    # Random Forest
    intent_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    intent_rf.fit(X_train, y_train)
    y_pred_rf = intent_rf.predict(X_test)
    intent_accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    # Gradient Boosting
    intent_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    intent_gb.fit(X_train, y_train)
    y_pred_gb = intent_gb.predict(X_test)
    intent_accuracy_gb = accuracy_score(y_test, y_pred_gb)
    
    # Select best model
    intent_accuracy = max(intent_accuracy_xgb, intent_accuracy_rf, intent_accuracy_gb)
    best_intent_model = intent_xgb if intent_accuracy_xgb == intent_accuracy else (intent_rf if intent_accuracy_rf == intent_accuracy else intent_gb)
    best_intent_name = 'XGBoost' if intent_accuracy_xgb == intent_accuracy else ('RandomForest' if intent_accuracy_rf == intent_accuracy else 'GradientBoosting')
    
    print(f"   [+] XGBoost Accuracy: {intent_accuracy_xgb:.2%}")
    print(f"   [+] RandomForest Accuracy: {intent_accuracy_rf:.2%}")
    print(f"   [+] GradientBoosting Accuracy: {intent_accuracy_gb:.2%}")
    print(f"   [+] Best Model: {best_intent_name} ({intent_accuracy:.2%})")
    print(f"   [+] Classes: {list(intent_encoder.classes_)}")
    
    joblib.dump(best_intent_model, MODEL_DIR / 'purchase_intent_model.pkl')
    joblib.dump(intent_encoder, MODEL_DIR / 'intent_encoder.pkl')
    
    # ========================================
    # MODEL 2: Loyalty Program Classification
    # ========================================
    print("\n5. Training Loyalty Program Models (Ensemble)...")
    
    y_loyalty = df['Customer_Loyalty_Program_Member'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_loyalty, test_size=0.2, random_state=42, stratify=y_loyalty
    )
    
    # XGBoost
    loyalty_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    loyalty_xgb.fit(X_train, y_train, verbose=False)
    y_pred_xgb = loyalty_xgb.predict(X_test)
    loyalty_accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    
    # RandomForest
    loyalty_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    loyalty_rf.fit(X_train, y_train)
    y_pred_rf = loyalty_rf.predict(X_test)
    loyalty_accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    # Gradient Boosting
    loyalty_gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    loyalty_gb.fit(X_train, y_train)
    y_pred_gb = loyalty_gb.predict(X_test)
    loyalty_accuracy_gb = accuracy_score(y_test, y_pred_gb)
    
    loyalty_accuracy = max(loyalty_accuracy_xgb, loyalty_accuracy_rf, loyalty_accuracy_gb)
    best_loyalty_model = loyalty_xgb if loyalty_accuracy_xgb == loyalty_accuracy else (loyalty_rf if loyalty_accuracy_rf == loyalty_accuracy else loyalty_gb)
    best_loyalty_name = 'XGBoost' if loyalty_accuracy_xgb == loyalty_accuracy else ('RandomForest' if loyalty_accuracy_rf == loyalty_accuracy else 'GradientBoosting')
    
    print(f"   [+] XGBoost Accuracy: {loyalty_accuracy_xgb:.2%}")
    print(f"   [+] RandomForest Accuracy: {loyalty_accuracy_rf:.2%}")
    print(f"   [+] GradientBoosting Accuracy: {loyalty_accuracy_gb:.2%}")
    print(f"   [+] Best Model: {best_loyalty_name} ({loyalty_accuracy:.2%})")
    
    joblib.dump(best_loyalty_model, MODEL_DIR / 'loyalty_model.pkl')
    
    # ========================================
    # MODEL 3: Purchase Amount Regression
    # ========================================
    print("\n6. Training Purchase Amount Models (Regression)...")
    
    y_amount = df['Purchase_Amount']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_amount, test_size=0.2, random_state=42
    )
    
    # XGBoost Regressor
    amount_xgb = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    amount_xgb.fit(X_train, y_train, verbose=False)
    y_pred_xgb = amount_xgb.predict(X_test)
    amount_r2_xgb = r2_score(y_test, y_pred_xgb)
    amount_rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    
    # RandomForest Regressor
    amount_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    amount_rf.fit(X_train, y_train)
    y_pred_rf = amount_rf.predict(X_test)
    amount_r2_rf = r2_score(y_test, y_pred_rf)
    amount_rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    # Gradient Boosting
    amount_gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    amount_gb.fit(X_train, y_train)
    y_pred_gb = amount_gb.predict(X_test)
    amount_r2_gb = r2_score(y_test, y_pred_gb)
    amount_rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    
    amount_r2 = max(amount_r2_xgb, amount_r2_rf, amount_r2_gb)
    best_amount_model = amount_xgb if amount_r2_xgb == amount_r2 else (amount_rf if amount_r2_rf == amount_r2 else amount_gb)
    best_amount_name = 'XGBoost' if amount_r2_xgb == amount_r2 else ('RandomForest' if amount_r2_rf == amount_r2 else 'GradientBoosting')
    
    print(f"   [+] XGBoost - R2: {amount_r2_xgb:.2%}, RMSE: ${amount_rmse_xgb:.2f}")
    print(f"   [+] RandomForest - R2: {amount_r2_rf:.2%}, RMSE: ${amount_rmse_rf:.2f}")
    print(f"   [+] GradientBoosting - R2: {amount_r2_gb:.2%}, RMSE: ${amount_rmse_gb:.2f}")
    print(f"   [+] Best Model: {best_amount_name} (R2: {amount_r2:.2%})")
    
    joblib.dump(best_amount_model, MODEL_DIR / 'amount_model.pkl')
    
    # ========================================
    # Additional Analytics Models
    # ========================================
    print("\n7. Training Additional Analytics Models...")
    
    # Satisfaction Prediction
    y_satisfaction = df['Customer_Satisfaction']
    X_train_sat, X_test_sat, y_train_sat, y_test_sat = train_test_split(
        X_scaled, y_satisfaction, test_size=0.2, random_state=42
    )
    
    satisfaction_model = XGBRegressor(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    satisfaction_model.fit(X_train_sat, y_train_sat, verbose=False)
    satisfaction_r2 = r2_score(y_test_sat, satisfaction_model.predict(X_test_sat))
    joblib.dump(satisfaction_model, MODEL_DIR / 'satisfaction_model.pkl')
    print(f"   [+] Customer Satisfaction Model (R2: {satisfaction_r2:.2%})")
    
    # Return Rate Prediction
    y_return = df['Return_Rate']
    X_train_ret, X_test_ret, y_train_ret, y_test_ret = train_test_split(
        X_scaled, y_return, test_size=0.2, random_state=42
    )
    
    return_model = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    return_model.fit(X_train_ret, y_train_ret, verbose=False)
    return_accuracy = accuracy_score(y_test_ret, return_model.predict(X_test_ret))
    joblib.dump(return_model, MODEL_DIR / 'return_rate_model.pkl')
    print(f"   [+] Return Rate Prediction Model (Accuracy: {return_accuracy:.2%})")
    
    # ========================================
    # Save Metadata
    # ========================================
    print("\n8. Saving model metadata...")
    
    metadata = {
        'feature_columns': feature_columns,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'intent_classes': list(intent_encoder.classes_),
        'models_used': {
            'purchase_intent': best_intent_name,
            'loyalty_program': best_loyalty_name,
            'purchase_amount': best_amount_name
        },
        'metrics': {
            'purchase_intent_accuracy': float(intent_accuracy),
            'loyalty_accuracy': float(loyalty_accuracy),
            'amount_r2_score': float(amount_r2),
            'amount_rmse': float(amount_rmse_xgb),
            'satisfaction_r2': float(satisfaction_r2),
            'return_accuracy': float(return_accuracy)
        },
        'dataset_info': {
            'total_samples': len(df),
            'features': len(feature_columns),
            'data_file': 'Ecommerce_Consumer_Behavior_Analysis_Data.csv'
        }
    }
    
    with open(MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   [+] Metadata saved")
    
    # ========================================
    # Advanced EDA Data
    # ========================================
    print("\n9. Generating Advanced EDA data...")
    
    eda_data = {
        'dataset_overview': {
            'total_records': len(df),
            'total_revenue': float(df['Purchase_Amount'].sum()),
            'avg_purchase': float(df['Purchase_Amount'].mean()),
            'median_purchase': float(df['Purchase_Amount'].median()),
        },
        'income_analysis': {
            'by_income': df.groupby('Income_Level')['Purchase_Amount'].agg(['mean', 'count']).to_dict(),
        },
        'loyalty_stats': {
            'members': int(df['Customer_Loyalty_Program_Member'].sum()),
            'non_members': int((~df['Customer_Loyalty_Program_Member']).sum()),
            'member_avg_spending': float(df[df['Customer_Loyalty_Program_Member']]['Purchase_Amount'].mean()),
            'non_member_avg_spending': float(df[~df['Customer_Loyalty_Program_Member']]['Purchase_Amount'].mean()),
            'member_satisfaction': float(df[df['Customer_Loyalty_Program_Member']]['Customer_Satisfaction'].mean()),
        },
        'purchase_intent_distribution': df['Purchase_Intent'].value_counts().to_dict(),
        'top_categories': df['Purchase_Category'].value_counts().head(10).to_dict(),
        'channel_performance': {
            'channel_revenue': df.groupby('Purchase_Channel')['Purchase_Amount'].mean().to_dict(),
            'channel_satisfaction': df.groupby('Purchase_Channel')['Customer_Satisfaction'].mean().to_dict(),
        },
        'discount_impact': {
            'with_discount': float(df[df['Discount_Used']]['Purchase_Amount'].mean()),
            'without_discount': float(df[~df['Discount_Used']]['Purchase_Amount'].mean()),
            'discount_penetration': float((df['Discount_Used'].sum() / len(df)) * 100),
        },
        'demographic_insights': {
            'avg_age': float(df['Age'].mean()),
            'gender_distribution': df['Gender'].value_counts().to_dict(),
            'education_distribution': df['Education_Level'].value_counts().to_dict(),
        },
        'satisfaction_metrics': {
            'avg_satisfaction': float(df['Customer_Satisfaction'].mean()),
            'satisfaction_by_intent': df.groupby('Purchase_Intent')['Customer_Satisfaction'].mean().to_dict(),
        },
        'correlation_analysis': {
            'brand_loyalty_vs_satisfaction': float(df['Brand_Loyalty'].corr(df['Customer_Satisfaction'])),
            'age_vs_amount': float(df['Age'].corr(df['Purchase_Amount'])),
            'time_to_decision_vs_satisfaction': float(df['Time_to_Decision'].corr(df['Customer_Satisfaction'])),
            'frequency_vs_satisfaction': float(df['Frequency_of_Purchase'].corr(df['Customer_Satisfaction'])),
        },
        'device_insights': {
            'by_device': df.groupby('Device_Used_for_Shopping')['Purchase_Amount'].agg(['mean', 'count']).to_dict(),
        }
    }
    
    with open(MODEL_DIR / 'eda_data.json', 'w') as f:
        json.dump(eda_data, f, indent=2, default=str)
    
    print("   [+] Advanced EDA data saved")
    
    print("\n" + "=" * 60)
    print("[OK] ADVANCED MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nModels saved in: {MODEL_DIR}")
    print("\nTrained Models:")
    print(f"  1. purchase_intent_model.pkl ({best_intent_name})")
    print(f"  2. loyalty_model.pkl ({best_loyalty_name})")
    print(f"  3. amount_model.pkl ({best_amount_name})")
    print(f"  4. satisfaction_model.pkl (XGBoost)")
    print(f"  5. return_rate_model.pkl (XGBoost)")
    print("\nPerformance Metrics:")
    print(f"  Purchase Intent Accuracy: {intent_accuracy:.2%}")
    print(f"  Loyalty Prediction Accuracy: {loyalty_accuracy:.2%}")
    print(f"  Amount Prediction R2: {amount_r2:.2%}")
    print(f"  Customer Satisfaction R2: {satisfaction_r2:.2%}")
    print(f"  Return Rate Accuracy: {return_accuracy:.2%}")
    
    return metadata

if __name__ == '__main__':
    try:
        metadata = train_models()
    except Exception as e:
        print(f"\n[ERROR] during training: {str(e)}")
        import traceback
        traceback.print_exc()
