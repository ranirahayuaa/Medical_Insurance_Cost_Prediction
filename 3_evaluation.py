import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Load data dan model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/insurance.csv')

# Encode
label_encoders = {}
categorical_cols = ['sex', 'smoker', 'region']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load model terbaik
best_model = joblib.load('models/best_model.pkl')

# ==================== PREDIKSI ====================
print("="*60)
print("EVALUASI MODEL")
print("="*60)

y_pred = best_model.predict(X_test_scaled)

# ==================== METRIK ====================
print("\nMETRIK EVALUASI:")
print("-"*40)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"RMSE: ${rmse:,.2f}")
print(f"MAE : ${mae:,.2f}")
print(f"MSE : ${mse:,.2f}")
print(f"R²  : {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# ==================== FEATURE IMPORTANCE ====================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importance = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance.to_string(index=False))

# ==================== SAVE EVALUATION TO JSON ====================
print("\n" + "="*60)
print("SAVE EVALUATION TO JSON")
print("="*60)

evaluation_results = {
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'Random Forest (Tuned)',
    'metrics': {
        'rmse': float(rmse),
        'mae': float(mae),
        'mse': float(mse),
        'r2': float(r2),
        'mape': float(mape)
    },
    'feature_importance': {
        row['Fitur']: float(row['Importance']) 
        for _, row in importance.iterrows()
    },
    'data_info': {
        'total_samples': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
}

# Simpan ke JSON
os.makedirs('results', exist_ok=True)
with open('results/evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print("✅ Evaluasi disimpan ke 'results/evaluation_results.json'")

# ==================== VISUALISASI ====================
print("\n" + "="*60)
print("MEMBUAT VISUALISASI")
print("="*60)

os.makedirs('results', exist_ok=True)

# 1. Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x='Importance', y='Fitur')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('results/feature_importance.png')
plt.show()

# 2. Residual Analysis
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Residual plot
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted Charges')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residual Plot')

# Histogram residual
sns.histplot(residuals, kde=True, ax=axes[1])
axes[1].set_title('Histogram Residual')
axes[1].set_xlabel('Residuals')

# Actual vs Predicted
axes[2].scatter(y_test, y_pred, alpha=0.5)
axes[2].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--')
axes[2].set_xlabel('Actual Charges')
axes[2].set_ylabel('Predicted Charges')
axes[2].set_title('Actual vs Predicted')

plt.tight_layout()
plt.savefig('results/residual_analysis.png')
plt.show()

print("Visualisasi disimpan di folder 'results/'")

# ==================== SUMMARY ====================
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)

print(f"""
📊 Model Performance Summary
{'='*40}

Best Model: Random Forest (Tuned)

Performance Metrics:
- R² Score  : {r2:.4f} ({r2*100:.2f}% of variance explained)
- RMSE      : ${rmse:,.2f}
- MAE       : ${mae:,.2f}
- MSE       : ${mse:,.2f}
- MAPE      : {mape:.2f}%

Feature Importance (Top 3):
1. {importance.iloc[0]['Fitur']}: {importance.iloc[0]['Importance']:.2%}
2. {importance.iloc[1]['Fitur']}: {importance.iloc[1]['Importance']:.2%}
3. {importance.iloc[2]['Fitur']}: {importance.iloc[2]['Importance']:.2%}

Data Split:
- Total Samples: {len(df)}
- Training Set: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)
- Test Set: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)

Files Saved:
- models/best_model.pkl
- models/best_model_info.json
- results/model_comparison.csv
- results/model_comparison.json
- results/evaluation_results.json
- results/feature_importance.png
- results/residual_analysis.png
""")

print("\n✅ EVALUASI SELESAI!")