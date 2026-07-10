import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def load_data():
    """Loads the application and credit record datasets."""
    print("Loading datasets...")
    try:
        app_df = pd.read_csv(config.DATASET_PATH)
        credit_df = pd.read_csv(config.CREDIT_RECORD_PATH)
        return app_df, credit_df
    except FileNotFoundError as e:
        print(f"Error: Dataset not found. Please ensure data is downloaded to {config.DATA_DIR}")
        raise e

def clean_data(app_df, credit_df):
    """Handles missing values, duplicates, and outliers."""
    print("Cleaning data...")
    
    # 1. Application Data Cleaning
    # Drop duplicates keeping the last entry based on ID
    app_df = app_df.drop_duplicates(subset=['ID'], keep='last')
    
    # Handle missing values
    # OCCUPATION_TYPE has missing values, fill with 'Unknown'
    app_df['OCCUPATION_TYPE'].fillna('Unknown', inplace=True)
    
    # Handle outliers/invalid entries
    # DAYS_EMPLOYED has a value 365243 which means pension/unemployed. Replace with 0 or NaN.
    # We will replace it with 0 to indicate no current employment days.
    app_df['DAYS_EMPLOYED'] = app_df['DAYS_EMPLOYED'].replace(365243, 0)
    
    # Convert DAYS_BIRTH and DAYS_EMPLOYED to positive years
    app_df['AGE_YEARS'] = np.abs(app_df['DAYS_BIRTH']) / 365.25
    app_df['YEARS_EMPLOYED'] = np.abs(app_df['DAYS_EMPLOYED']) / 365.25
    
    # Drop the original DAYS_ columns
    app_df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED'], inplace=True)
    
    # 2. Credit Data Cleaning
    # No major missing values in credit data, but we need to create the Target Variable
    
    return app_df, credit_df

def create_target(credit_df):
    """
    Creates the target variable 'Target' (1 for Approved/Good, 0 for Rejected/Bad)
    based on payment history.
    """
    print("Creating target variable based on payment history...")
    # Status codes: C, X, 0, 1, 2, 3, 4, 5
    # Let's consider 0, 1, 2, 3, 4, 5 as past due. 
    # Usually, a user is considered 'Bad' if they have a status of 2, 3, 4, 5 (>= 60 days past due).
    # We will mark users who have EVER had a status of '2', '3', '4', '5' as Bad (0: Rejected).
    # Users who only have 'C', 'X', '0', '1' are Good (1: Approved).
    
    bad_statuses = ['2', '3', '4', '5']
    credit_df['Is_Bad'] = credit_df['STATUS'].apply(lambda x: 1 if x in bad_statuses else 0)
    
    # Group by ID to see if the user ever had a bad status
    user_status = credit_df.groupby('ID')['Is_Bad'].max().reset_index()
    
    # Target = 1 (Approved/Good risk), 0 (Rejected/Bad risk)
    user_status['Target'] = 1 - user_status['Is_Bad']
    user_status.drop(columns=['Is_Bad'], inplace=True)
    
    return user_status

def preprocess_pipeline():
    """Executes the full preprocessing pipeline."""
    app_df, credit_df = load_data()
    app_df, credit_df = clean_data(app_df, credit_df)
    
    target_df = create_target(credit_df)
    
    # Merge application data with target
    final_df = pd.merge(app_df, target_df, on='ID', how='inner')
    print(f"Preprocessed data shape: {final_df.shape}")
    
    return final_df

if __name__ == "__main__":
    df = preprocess_pipeline()
    print(df.head())
    print("Target distribution:")
    print(df['Target'].value_counts(normalize=True))
