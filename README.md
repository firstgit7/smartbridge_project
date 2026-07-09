# AI Powered Credit Card Approval Prediction System

## Overview
An end-to-end Machine Learning project to automate the credit card approval process. The system analyzes applicant financial and demographic information to predict whether an application should be approved or rejected.

## Features
- **Data Pipeline**: Handles missing values, encodes categorical variables, and generates new financial features (e.g., Financial Stability Score).
- **ML Models**: Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- **REST API**: Flask backend providing predictions via JSON.
- **Web Interface**: Modern UI with a dark-mode banking theme, responsive design, and dynamic forms.
- **Dashboard**: Real-time analytics of predictions, approval rates, and risk distributions using Chart.js.

## Technologies Used
- **Backend**: Python, Flask, SQLite
- **Machine Learning**: Scikit-learn, XGBoost, Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**:
   Run the dataset downloader to fetch the Kaggle dataset.
   ```bash
   python download_dataset.py
   ```

3. **Train the Model**:
   Execute the training pipeline to generate the models and encoders.
   ```bash
   python src/train_model.py
   ```

4. **Run the Flask App**:
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` in your web browser.

## Scenarios Implemented
- **Automated Screening**: Real-time prediction form for applicants.
- **Dashboard**: High-risk applicant identification and batch analytics.

## Folder Structure
```
CreditCardApproval/
├── app.py
├── config.py
├── requirements.txt
├── download_dataset.py
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
├── templates/
│   ├── index.html
│   ├── prediction.html
│   └── dashboard.html
├── static/
│   ├── css/style.css
│   └── js/main.js
├── model/ (Generated after training)
└── dataset/ (Downloaded via kagglehub)
```
