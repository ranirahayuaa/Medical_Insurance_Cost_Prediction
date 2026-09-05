import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Coba import XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] XGBoost tidak terinstall, akan dilewati")

# Load data hasil preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load kembali data (atau bisa import dari preprocessing)
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

# ==================== DEFINISI MODEL ====================
print("="*60)
print("TRAINING MODELS")
print("="*60)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Tambahkan XGBoost jika tersedia
if HAS_XGBOOST:
    models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)

print(f"\nTotal model: {len(models)}")
print(f"Model yang akan dilatih: {', '.join(models.keys())}")

# ==================== TRAINING & EVALUASI ====================
results = {
    'Model': [],
    'RMSE Train': [],
    'RMSE Test': [],
    'R2 Train': [],
    'R2 Test': [],
    'CV Score': [],
    'Status': []
}

print("\nTraining models...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    try:
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Prediksi
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Metrik
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        
        # Cross validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Simpan
        results['Model'].append(name)
        results['RMSE Train'].append(rmse_train)
        results['RMSE Test'].append(rmse_test)
        results['R2 Train'].append(r2_train)
        results['R2 Test'].append(r2_test)
        results['CV Score'].append(cv_scores.mean())
        results['Status'].append('Success')
        
        print(f"  [OK] R2 Test: {r2_test:.4f} | RMSE Test: ${rmse_test:,.2f}")
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Model'].append(name)
        results['RMSE Train'].append(np.nan)
        results['RMSE Test'].append(np.nan)
        results['R2 Train'].append(np.nan)
        results['R2 Test'].append(np.nan)
        results['CV Score'].append(np.nan)
        results['Status'].append(f'Failed: {str(e)}')

# ==================== HASIL ====================
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("HASIL PERBANDINGAN MODEL")
print("="*60)
print(results_df.to_string(index=False))

# Simpan hasil ke CSV
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/model_comparison.csv', index=False)

# ==================== SAVE RESULTS TO JSON ====================
print("\n" + "="*60)
print("SAVE RESULTS TO JSON")
print("="*60)

# Konversi DataFrame ke dictionary untuk JSON
results_dict = {
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_models': len(results_df),
    'models': []
}

for idx, row in results_df.iterrows():
    model_data = {
        'model_name': row['Model'],
        'metrics': {
            'rmse_train': float(row['RMSE Train']) if not pd.isna(row['RMSE Train']) else None,
            'rmse_test': float(row['RMSE Test']) if not pd.isna(row['RMSE Test']) else None,
            'r2_train': float(row['R2 Train']) if not pd.isna(row['R2 Train']) else None,
            'r2_test': float(row['R2 Test']) if not pd.isna(row['R2 Test']) else None,
            'cv_score': float(row['CV Score']) if not pd.isna(row['CV Score']) else None
        },
        'status': row['Status']
    }
    results_dict['models'].append(model_data)

# Simpan ke JSON
with open('results/model_comparison.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("[OK] Hasil disimpan ke 'results/model_comparison.json'")

# ==================== HYPERPARAMETER TUNING ====================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("="*60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# Evaluasi model terbaik
y_pred = best_model.predict(X_test_scaled)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred))
r2_best = r2_score(y_test, y_pred)
mae_best = mean_absolute_error(y_test, y_pred)

print(f"\nModel terbaik setelah tuning:")
print(f"RMSE Test: ${rmse_best:,.2f}")
print(f"R2 Test: {r2_best:.4f}")
print(f"MAE Test: ${mae_best:,.2f}")

# ==================== SAVE MODEL ====================
print("\n" + "="*60)
print("SAVE MODEL")
print("="*60)

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.pkl')
print("Model terbaik disimpan di 'models/best_model.pkl'")

# ==================== SAVE BEST MODEL INFO TO JSON ====================
best_model_info = {
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'best_model': {
        'name': 'Random Forest (Tuned)',
        'parameters': grid_search.best_params_,
        'performance': {
            'rmse_test': float(rmse_best),
            'r2_test': float(r2_best),
            'mae_test': float(mae_best),
            'cv_score': float(grid_search.best_score_)
        },
        'feature_importance': {}
    }
}

# Tambahkan feature importance
for feature, importance in zip(X.columns, best_model.feature_importances_):
    best_model_info['best_model']['feature_importance'][feature] = float(importance)

# Simpan ke JSON
with open('models/best_model_info.json', 'w') as f:
    json.dump(best_model_info, f, indent=2)

print("[OK] Informasi model terbaik disimpan ke 'models/best_model_info.json'")

print("\n[OK] TRAINING SELESAI!")