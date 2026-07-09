import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def generate_financial_features(df):
    """Generates new financial features."""
    print("Generating new features...")
    
    # Avoid division by zero
    epsilon = 1e-6
    
    # Income per Family Member
    df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + epsilon)
    
    # Employment Stability (Years Employed / Age)
    df['EMPLOYMENT_STABILITY'] = df['YEARS_EMPLOYED'] / (df['AGE_YEARS'] + epsilon)
    
    # We don't have existing loan amount, but we can simulate a Debt Ratio or similar
    # based on property/car ownership as proxies for financial stability or loan burden.
    # Let's create a Financial Stability Score (higher is better)
    # Score increases with income, property ownership, and age, decreases with children
    
    df['FINANCIAL_STABILITY_SCORE'] = (
        np.log1p(df['AMT_INCOME_TOTAL']) + 
        df['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0}) * 2 + 
        df['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0}) * 1.5 + 
        df['AGE_YEARS'] * 0.1 -
        df['CNT_CHILDREN'] * 0.5
    )
    
    return df

def encode_features(df):
    """Applies Label Encoding and One Hot Encoding."""
    print("Encoding features...")
    
    df = df.copy()
    
    # Binary Categorical Variables -> Label Encoding or binary mapping
    binary_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    label_encoders = {}
    
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    # Multi-class Categorical Variables -> One Hot Encoding
    # We will use pandas get_dummies for simplicity, but in production, OneHotEncoder is better
    # to handle unknown categories during inference. Here we use OneHotEncoder for the pipeline.
    
    from sklearn.preprocessing import OneHotEncoder
    
    categorical_cols = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
                        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    
    # We will save the encoder and expected columns for inference
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Drop original categorical columns and concatenate encoded ones
    df = df.drop(columns=categorical_cols)
    # Reset index before concatenation to avoid NaNs
    df.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    
    # Save the encoder for inference
    joblib.dump({'label_encoders': label_encoders, 'one_hot_encoder': encoder, 'cat_cols': categorical_cols}, config.ENCODER_PATH)
    print(f"Encoders saved to {config.ENCODER_PATH}")
    
    return df

def feature_engineering_pipeline(df):
    """Executes the full feature engineering pipeline."""
    df = generate_financial_features(df)
    
    # Drop ID as it's not a predictive feature
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    df = encode_features(df)
    
    # Separate features and target
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved to {config.SCALER_PATH}")
    
    # Convert scaled arrays back to DataFrames to keep column names (useful for SHAP and feature importance)
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from preprocessing import preprocess_pipeline
    df = preprocess_pipeline()
    X_train, X_test, y_train, y_test = feature_engineering_pipeline(df)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
