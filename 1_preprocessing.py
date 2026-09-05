import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ==================== LOAD DATA ====================
print("="*60)
print("LOADING DATA")
print("="*60)

df = pd.read_csv('data/insurance.csv')
print(f"Jumlah data: {len(df)}")
print(f"Kolom: {df.columns.tolist()}")
print("\n5 data pertama:")
print(df.head())

# ==================== CEK DATA ====================
print("\n" + "="*60)
print("CEK DATA")
print("="*60)

print("\nInfo dataset:")
print(df.info())

print("\nStatistik deskriptif:")
print(df.describe())

print(f"\nMissing values: {df.isnull().sum().sum()}")

# ==================== ENCODE KATEGORIKAL ====================
print("\n" + "="*60)
print("ENCODE KATEGORIKAL")
print("="*60)

label_encoders = {}
categorical_cols = ['sex', 'smoker', 'region']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"\n{col}:")
    for cls, val in zip(le.classes_, le.transform(le.classes_)):
        print(f"  {cls} -> {val}")

# ==================== SPLIT DATA ====================
print("\n" + "="*60)
print("SPLIT DATA")
print("="*60)

X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# ==================== STANDARISASI ====================
print("\n" + "="*60)
print("STANDARISASI DATA")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data berhasil distandarisasi")

# ==================== SAVE PREPROCESSOR ====================
print("\n" + "="*60)
print("SAVE PREPROCESSOR")
print("="*60)

os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')

print("Preprocessor disimpan di folder 'models/'")

print("\n✅ PREPROCESSING SELESAI!")