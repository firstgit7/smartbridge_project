import joblib
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# Load models and encoders globally
best_model = None
scaler = None
encoders = None

def load_resources():
    """Load model, scaler, and encoders."""
    global best_model, scaler, encoders
    
    if best_model is None:
        try:
            best_model = joblib.load(config.BEST_MODEL_PATH)
            scaler = joblib.load(config.SCALER_PATH)
            encoders = joblib.load(config.ENCODER_PATH)
            print("Model and transformers loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Make sure you have trained the model first.")

def preprocess_input(data):
    """Preprocess the input data exactly as during training."""
    df = pd.DataFrame([data])
    
    # Generate financial features
    epsilon = 1e-6
    df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + epsilon)
    df['EMPLOYMENT_STABILITY'] = df['YEARS_EMPLOYED'] / (df['AGE_YEARS'] + epsilon)
    
    df['FINANCIAL_STABILITY_SCORE'] = (
        np.log1p(df['AMT_INCOME_TOTAL']) + 
        df['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0}) * 2 + 
        df['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0}) * 1.5 + 
        df['AGE_YEARS'] * 0.1 -
        df['CNT_CHILDREN'] * 0.5
    )
    
    # Label encode binary cols
    binary_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    for col in binary_cols:
        le = encoders['label_encoders'][col]
        # Handle unseen categories gracefully
        if data[col] in le.classes_:
            df[col] = le.transform([data[col]])
        else:
            df[col] = 0 # Default to 0 if unseen
            
    # One Hot Encode categorical cols
    cat_cols = encoders['cat_cols']
    one_hot_encoder = encoders['one_hot_encoder']
    
    encoded_features = one_hot_encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(cat_cols))
    
    df = df.drop(columns=cat_cols)
    df.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    return pd.DataFrame(df_scaled, columns=df.columns)

def predict_approval(data):
    """Make a prediction based on input data."""
    load_resources()
    
    processed_data = preprocess_input(data)
    
    prediction = best_model.predict(processed_data)[0]
    probability = best_model.predict_proba(processed_data)[0][1] if hasattr(best_model, "predict_proba") else prediction
    
    # The model tends to predict high probabilities due to the inherent data distribution.
    # We will enforce a strict threshold (e.g., 90%) for approval to minimize risk.
    STRICT_THRESHOLD = 0.90
    
    is_approved = probability >= STRICT_THRESHOLD
    result = "Approved" if is_approved else "Rejected"
    
    # Adjust risk levels based on this new strict threshold
    risk_level = "High" if probability < 0.85 else "Medium" if probability < 0.95 else "Low"
    
    return {
        "prediction": result,
        "approval_probability": round(float(probability) * 100, 2),
        "risk_level": risk_level,
        "confidence_score": round(float(probability if is_approved else 1-probability) * 100, 2)
    }

if __name__ == "__main__":
    # Test prediction
    sample_data = {
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'Y',
        'FLAG_OWN_REALTY': 'Y',
        'CNT_CHILDREN': 0,
        'AMT_INCOME_TOTAL': 200000,
        'NAME_INCOME_TYPE': 'Working',
        'NAME_EDUCATION_TYPE': 'Higher education',
        'NAME_FAMILY_STATUS': 'Married',
        'NAME_HOUSING_TYPE': 'House / apartment',
        'FLAG_MOBIL': 1,
        'FLAG_WORK_PHONE': 1,
        'FLAG_PHONE': 0,
        'FLAG_EMAIL': 0,
        'OCCUPATION_TYPE': 'Core staff',
        'CNT_FAM_MEMBERS': 2.0,
        'AGE_YEARS': 35.0,
        'YEARS_EMPLOYED': 10.0
    }
    
    print(predict_approval(sample_data))
