import json
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.predict import predict_approval

print("========================================")
print("TEST CASE 1: High Income, Low Risk (Ideal Candidate)")
print("========================================")
good_candidate = {
    'CODE_GENDER': 'F',
    'FLAG_OWN_CAR': 'Y',
    'FLAG_OWN_REALTY': 'Y',
    'CNT_CHILDREN': 0,
    'AMT_INCOME_TOTAL': 150000,
    'NAME_INCOME_TYPE': 'Commercial associate',
    'NAME_EDUCATION_TYPE': 'Higher education',
    'NAME_FAMILY_STATUS': 'Married',
    'NAME_HOUSING_TYPE': 'House / apartment',
    'FLAG_MOBIL': 1,
    'FLAG_WORK_PHONE': 1,
    'FLAG_PHONE': 1,
    'FLAG_EMAIL': 1,
    'OCCUPATION_TYPE': 'Managers',
    'CNT_FAM_MEMBERS': 2.0,
    'AGE_YEARS': 45.0,
    'YEARS_EMPLOYED': 15.0
}

result_good = predict_approval(good_candidate)
print(json.dumps(result_good, indent=2))
print("\n")


print("========================================")
print("TEST CASE 2: Low Income, High Risk Candidate")
print("========================================")
bad_candidate = {
    'CODE_GENDER': 'M',
    'FLAG_OWN_CAR': 'N',
    'FLAG_OWN_REALTY': 'N',
    'CNT_CHILDREN': 4,
    'AMT_INCOME_TOTAL': 15000,
    'NAME_INCOME_TYPE': 'Pensioner',
    'NAME_EDUCATION_TYPE': 'Lower secondary',
    'NAME_FAMILY_STATUS': 'Single / not married',
    'NAME_HOUSING_TYPE': 'Rented apartment',
    'FLAG_MOBIL': 1,
    'FLAG_WORK_PHONE': 0,
    'FLAG_PHONE': 0,
    'FLAG_EMAIL': 0,
    'OCCUPATION_TYPE': 'Laborers',
    'CNT_FAM_MEMBERS': 5.0,
    'AGE_YEARS': 22.0,
    'YEARS_EMPLOYED': 0.5
}

result_bad = predict_approval(bad_candidate)
print(json.dumps(result_bad, indent=2))
print("\n")
