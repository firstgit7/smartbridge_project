from flask import Flask, render_template, request, jsonify
import os
import sqlite3
from datetime import datetime

from config import config
from src.predict import predict_approval

app = Flask(__name__)
app.config.from_object(config)

# Initialize Database
def init_db():
    db_path = os.path.join(config.BASE_DIR, 'app.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            age REAL,
            income REAL,
            employment_years REAL,
            prediction TEXT,
            probability REAL,
            risk_level TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction', methods=['GET'])
def prediction_page():
    return render_template('prediction.html')

@app.route('/dashboard')
def dashboard():
    # Fetch prediction history from DB for dashboard
    db_path = os.path.join(config.BASE_DIR, 'app.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT prediction, probability, risk_level, income FROM predictions ORDER BY timestamp DESC LIMIT 100')
    rows = c.fetchall()
    conn.close()
    
    # Process data for charts
    total_preds = len(rows)
    approved = sum(1 for r in rows if r[0] == 'Approved')
    rejected = total_preds - approved
    
    approval_rate = round((approved / total_preds * 100) if total_preds > 0 else 0, 1)
    
    risk_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    for r in rows:
        if r[2] in risk_counts:
            risk_counts[r[2]] += 1
            
    stats = {
        'total': total_preds,
        'approval_rate': approval_rate,
        'rejected_rate': 100 - approval_rate if total_preds > 0 else 0,
        'risk_distribution': risk_counts,
        'recent': rows[:10]
    }
    
    return render_template('dashboard.html', stats=stats)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Calculate family members based on marital status
        num_children = int(data.get('children', 0))
        marital_status = data.get('family_status', 'Married')
        
        # If married or civil marriage, assume 2 adults, else 1 adult
        adults = 2 if marital_status in ['Married', 'Civil marriage'] else 1
        cnt_fam_members = float(num_children + adults)
        
        # Cast inputs correctly
        input_data = {
            'CODE_GENDER': data.get('gender', 'M'),
            'FLAG_OWN_CAR': data.get('car', 'N'),
            'FLAG_OWN_REALTY': data.get('realty', 'Y'),
            'CNT_CHILDREN': num_children,
            'AMT_INCOME_TOTAL': float(data.get('income', 0)),
            'NAME_INCOME_TYPE': data.get('income_type', 'Working'),
            'NAME_EDUCATION_TYPE': data.get('education', 'Secondary / secondary special'),
            'NAME_FAMILY_STATUS': marital_status,
            'NAME_HOUSING_TYPE': data.get('housing_type', 'House / apartment'),
            'FLAG_MOBIL': 1, # Almost everyone has a mobile, we can leave as default
            'FLAG_WORK_PHONE': int(data.get('work_phone', 0)),
            'FLAG_PHONE': int(data.get('phone', 0)),
            'FLAG_EMAIL': int(data.get('email', 0)),
            'OCCUPATION_TYPE': data.get('occupation', 'Unknown'),
            'CNT_FAM_MEMBERS': cnt_fam_members,
            'AGE_YEARS': float(data.get('age', 30)),
            'YEARS_EMPLOYED': float(data.get('years_employed', 0))
        }
        
        result = predict_approval(input_data)
        
        # Log to DB
        db_path = os.path.join(config.BASE_DIR, 'app.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO predictions (timestamp, age, income, employment_years, prediction, probability, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(), input_data['AGE_YEARS'], input_data['AMT_INCOME_TOTAL'], 
            input_data['YEARS_EMPLOYED'], result['prediction'], result['approval_probability'], 
            result['risk_level']
        ))
        conn.commit()
        conn.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
