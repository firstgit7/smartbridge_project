import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline

def evaluate_model(y_true, y_pred, y_prob=None):
    """Calculates evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_prob)
    return metrics

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Trains multiple models, evaluates them, and returns the best one."""
    print("Training models...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=config.RANDOM_STATE, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=config.RANDOM_STATE)
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = evaluate_model(y_test, y_pred, y_prob)
        results[name] = metrics
        
        print(f"{name} Metrics: {metrics}")
        
        # We will use ROC AUC as the primary metric for best model selection if available, else F1
        score_to_compare = metrics.get('ROC AUC', metrics.get('F1 Score', 0))
        
        if score_to_compare > best_score:
            best_score = score_to_compare
            best_model = model
            best_model_name = name

    print("-" * 30)
    print("Model Comparison Results:")
    results_df = pd.DataFrame(results).T
    print(results_df)
    print("-" * 30)
    print(f"Best Model: {best_model_name} with score: {best_score:.4f}")
    
    return best_model, best_model_name, results_df

def run_training_pipeline():
    """Runs the full data loading, processing, and training pipeline."""
    df = preprocess_pipeline()
    X_train, X_test, y_train, y_test = feature_engineering_pipeline(df)
    
    # Handle highly imbalanced dataset using SMOTE
    print(f"Original class distribution in training set:\n{y_train.value_counts(normalize=True)}")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled class distribution:\n{y_train_resampled.value_counts(normalize=True)}")
    
    best_model, best_name, results = train_and_evaluate(X_train_resampled, X_test, y_train_resampled, y_test)
    
    # Save the best model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    
    # Save model metadata
    model_info = {
        'name': best_name,
        'features': list(X_train.columns),
        'metrics': results.loc[best_name].to_dict()
    }
    joblib.dump(model_info, os.path.join(config.MODEL_DIR, 'model_info.pkl'))
    
    print(f"Best model '{best_name}' saved to {config.BEST_MODEL_PATH}")

if __name__ == "__main__":
    run_training_pipeline()
