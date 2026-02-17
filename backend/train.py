import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent
MODEL_DIR = ROOT_DIR / 'model'
DATA_PATH = ROOT_DIR.parent / 'data' / 'dataset.csv'

def clean_purchase_amount(amount_str):
    """Clean purchase amount string to float"""
    if isinstance(amount_str, str):
        return float(amount_str.replace('$', '').replace(',', '').strip())
    return float(amount_str)

def train_models():
    print("=" * 50)
    print("Starting ML Model Training Pipeline")
    print("=" * 50)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Total samples: {len(df)}")
    
    # Data Cleaning
    print("\n2. Cleaning data...")
    df['Purchase_Amount'] = df['Purchase_Amount'].apply(clean_purchase_amount)
    
    # Handle missing values
    df['Social_Media_Influence'].fillna('None', inplace=True)
    df['Engagement_with_Ads'].fillna('None', inplace=True)
    
    print("   ✓ Missing values handled")
    print("   ✓ Purchase amount cleaned")
    
    # Feature Engineering
    print("\n3. Feature engineering...")
    
    # Define features for modeling (excluding target and ID columns)
    feature_columns = [
        'Age', 'Gender', 'Income_Level', 'Marital_Status', 'Education_Level',
        'Occupation', 'Purchase_Category', 'Frequency_of_Purchase',
        'Purchase_Channel', 'Brand_Loyalty', 'Product_Rating',
        'Time_Spent_on_Product_Research(hours)', 'Social_Media_Influence',
        'Discount_Sensitivity', 'Return_Rate', 'Customer_Satisfaction',
        'Engagement_with_Ads', 'Device_Used_for_Shopping', 'Payment_Method',
        'Time_to_Decision'
    ]
    
    # Separate numerical and categorical features
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
    
    print(f"   ✓ Encoded {len(categorical_features)} categorical features")
    
    # Save encoders
    joblib.dump(label_encoders, MODEL_DIR / 'label_encoders.pkl')
    print("   ✓ Label encoders saved")
    
    # Prepare features
    X = df_encoded[feature_columns]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
    print("   ✓ Feature scaler saved")
    
    # ========================================
    # MODEL 1: Purchase Intent Classification
    # ========================================
    print("\n4. Training Purchase Intent Model (Classification)...")
    
    y_intent = df['Purchase_Intent']
    intent_encoder = LabelEncoder()
    y_intent_encoded = intent_encoder.fit_transform(y_intent)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_intent_encoded, test_size=0.2, random_state=42
    )
    
    intent_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    intent_model.fit(X_train, y_train)
    
    y_pred = intent_model.predict(X_test)
    intent_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✓ Purchase Intent Accuracy: {intent_accuracy:.2%}")
    print(f"   ✓ Classes: {list(intent_encoder.classes_)}")
    
    joblib.dump(intent_model, MODEL_DIR / 'purchase_intent_model.pkl')
    joblib.dump(intent_encoder, MODEL_DIR / 'intent_encoder.pkl')
    
    # ========================================
    # MODEL 2: Loyalty Program Classification
    # ========================================
    print("\n5. Training Loyalty Program Model (Classification)...")
    
    y_loyalty = df['Customer_Loyalty_Program_Member'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_loyalty, test_size=0.2, random_state=42
    )
    
    loyalty_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    loyalty_model.fit(X_train, y_train)
    
    y_pred = loyalty_model.predict(X_test)
    loyalty_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✓ Loyalty Program Accuracy: {loyalty_accuracy:.2%}")
    
    joblib.dump(loyalty_model, MODEL_DIR / 'loyalty_model.pkl')
    
    # ========================================
    # MODEL 3: Purchase Amount Regression
    # ========================================
    print("\n6. Training Purchase Amount Model (Regression)...")
    
    y_amount = df['Purchase_Amount']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_amount, test_size=0.2, random_state=42
    )
    
    amount_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    amount_model.fit(X_train, y_train)
    
    y_pred = amount_model.predict(X_test)
    amount_r2 = r2_score(y_test, y_pred)
    amount_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"   ✓ Purchase Amount R² Score: {amount_r2:.2%}")
    print(f"   ✓ Purchase Amount RMSE: ${amount_rmse:.2f}")
    
    joblib.dump(amount_model, MODEL_DIR / 'amount_model.pkl')
    
    # ========================================
    # Save Metadata
    # ========================================
    print("\n7. Saving model metadata...")
    
    metadata = {
        'feature_columns': feature_columns,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'intent_classes': list(intent_encoder.classes_),
        'metrics': {
            'purchase_intent_accuracy': float(intent_accuracy),
            'loyalty_accuracy': float(loyalty_accuracy),
            'amount_r2_score': float(amount_r2),
            'amount_rmse': float(amount_rmse)
        },
        'dataset_info': {
            'total_samples': len(df),
            'features': len(feature_columns)
        }
    }
    
    with open(MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   ✓ Metadata saved")
    
    # ========================================
    # Generate EDA Data
    # ========================================
    print("\n8. Generating EDA data...")
    
    eda_data = {
        'income_vs_spending': df.groupby('Income_Level')['Purchase_Amount'].mean().to_dict(),
        'loyalty_stats': {
            'members': int(df['Customer_Loyalty_Program_Member'].sum()),
            'non_members': int((~df['Customer_Loyalty_Program_Member']).sum()),
            'member_avg_spending': float(df[df['Customer_Loyalty_Program_Member']]['Purchase_Amount'].mean()),
            'non_member_avg_spending': float(df[~df['Customer_Loyalty_Program_Member']]['Purchase_Amount'].mean())
        },
        'purchase_intent_distribution': df['Purchase_Intent'].value_counts().to_dict(),
        'top_categories': df['Purchase_Category'].value_counts().head(5).to_dict(),
        'discount_impact': {
            'with_discount': float(df[df['Discount_Used']]['Purchase_Amount'].mean()),
            'without_discount': float(df[~df['Discount_Used']]['Purchase_Amount'].mean())
        },
        'correlation_top_features': {
            'brand_loyalty_vs_satisfaction': float(df['Brand_Loyalty'].corr(df['Customer_Satisfaction'])),
            'age_vs_amount': float(df['Age'].corr(df['Purchase_Amount']))
        }
    }
    
    with open(MODEL_DIR / 'eda_data.json', 'w') as f:
        json.dump(eda_data, f, indent=2)
    
    print("   ✓ EDA data saved")
    
    print("\n" + "=" * 50)
    print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"\nModels saved in: {MODEL_DIR}")
    print("\nTrained Models:")
    print("  1. purchase_intent_model.pkl")
    print("  2. loyalty_model.pkl")
    print("  3. amount_model.pkl")
    print("\nMetrics Summary:")
    print(f"  Purchase Intent Accuracy: {intent_accuracy:.2%}")
    print(f"  Loyalty Prediction Accuracy: {loyalty_accuracy:.2%}")
    print(f"  Amount Prediction R²: {amount_r2:.2%}")
    
    return metadata

if __name__ == '__main__':
    try:
        metadata = train_models()
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
