"""
PROJECT: Medical Insurance Prediction
Run semua proses dari awal sampai akhir
"""

import json
import os
import pandas as pd
import sys

print("="*60)
print("MEDICAL INSURANCE PREDICTION")
print("="*60)

# ==================== STEP 1: PREPROCESSING ====================
print("\n" + "="*60)
print("STEP 1: PREPROCESSING DATA")
print("="*60)

with open('1_preprocessing.py', 'r', encoding='utf-8') as file:
    exec(file.read())

# ==================== STEP 2: TRAINING ====================
print("\n" + "="*60)
print("STEP 2: TRAINING MODELS")
print("="*60)

with open('2_training.py', 'r', encoding='utf-8') as file:
    exec(file.read())

# ==================== STEP 3: EVALUASI ====================
print("\n" + "="*60)
print("STEP 3: EVALUASI MODEL")
print("="*60)

with open('3_evaluation.py', 'r', encoding='utf-8') as file:
    exec(file.read())

# ==================== STEP 4: PREDIKSI ====================
print("\n" + "="*60)
print("STEP 4: PREDIKSI")
print("="*60)

with open('4_prediction.py', 'r', encoding='utf-8') as file:
    exec(file.read())

# ==================== FINAL SUMMARY ====================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

# Load hasil dari JSON
try:
    with open('results/evaluation_results.json', 'r') as f:
        eval_results = json.load(f)
    
    with open('models/best_model_info.json', 'r') as f:
        best_model_info = json.load(f)
    
    print(f"""
PROJECT COMPLETED SUCCESSFULLY!

Best Model: {best_model_info['best_model']['name']}
Model Parameters: {best_model_info['best_model']['parameters']}

Performance Metrics:
- R2 Score  : {eval_results['metrics']['r2']:.4f}
- RMSE      : ${eval_results['metrics']['rmse']:,.2f}
- MAE       : ${eval_results['metrics']['mae']:,.2f}
- MAPE      : {eval_results['metrics']['mape']:.2f}%

Top 3 Features:
1. {list(eval_results['feature_importance'].keys())[0]}: {list(eval_results['feature_importance'].values())[0]:.2%}
2. {list(eval_results['feature_importance'].keys())[1]}: {list(eval_results['feature_importance'].values())[1]:.2%}
3. {list(eval_results['feature_importance'].keys())[2]}: {list(eval_results['feature_importance'].values())[2]:.2%}

Files Saved:
models/
   |- best_model.pkl
   |- best_model_info.json
   |- scaler.pkl
   |- label_encoders.pkl

results/
   |- model_comparison.csv
   |- model_comparison.json
   |- evaluation_results.json
   |- prediction_examples.csv
   |- feature_importance.png
   |- residual_analysis.png

All files are ready!
""")

except Exception as e:
    print(f"Error loading summary: {e}")

print("\n" + "="*60)
print("ALL PROCESSES COMPLETED!")
print("="*60)