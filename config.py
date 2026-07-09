import os

class Config:
    # Directories
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    
    # Files
    DATASET_PATH = os.path.join(DATA_DIR, 'application_record.csv')
    CREDIT_RECORD_PATH = os.path.join(DATA_DIR, 'credit_record.csv')
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
    ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Database (for Flask app)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

config = Config()
