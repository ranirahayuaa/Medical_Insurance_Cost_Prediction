import pandas as pd
import numpy as np
import joblib
import json
import os

class InsurancePredictor:
    def __init__(self):
        # Load model dan preprocessor
        self.model = joblib.load('models/best_model.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.label_encoders = joblib.load('models/label_encoders.pkl')
        
        # Load model info
        if os.path.exists('models/best_model_info.json'):
            with open('models/best_model_info.json', 'r') as f:
                self.model_info = json.load(f)
        else:
            self.model_info = None
            
        print("✅ Model dan preprocessor berhasil dimuat!")
        if self.model_info:
            print(f"   Model: {self.model_info['best_model']['name']}")
            print(f"   R² Score: {self.model_info['best_model']['performance']['r2_test']:.4f}")
    
    def predict(self, age, sex, bmi, children, smoker, region):
        """
        Prediksi biaya asuransi
        
        Parameters:
        - age: int
        - sex: 'male' atau 'female'
        - bmi: float
        - children: int
        - smoker: 'yes' atau 'no'
        - region: 'southwest', 'southeast', 'northwest', 'northeast'
        
        Returns:
        - float: prediksi biaya
        """
        # Encode
        sex_encoded = self.label_encoders['sex'].transform([sex])[0]
        smoker_encoded = self.label_encoders['smoker'].transform([smoker])[0]
        region_encoded = self.label_encoders['region'].transform([region])[0]
        
        # Buat array
        data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
        
        # Standarisasi
        data_scaled = self.scaler.transform(data)
        
        # Prediksi
        prediction = self.model.predict(data_scaled)[0]
        
        return prediction
    
    def predict_with_details(self, age, sex, bmi, children, smoker, region):
        """Prediksi dengan tampilan detail"""
        prediction = self.predict(age, sex, bmi, children, smoker, region)
        
        print("\n" + "="*60)
        print("HASIL PREDIKSI ASURANSI KESEHATAN")
        print("="*60)
        print(f"Usia          : {age} tahun")
        print(f"Jenis Kelamin : {sex}")
        print(f"BMI           : {bmi}")
        print(f"Anak          : {children} orang")
        print(f"Perokok       : {smoker}")
        print(f"Wilayah       : {region}")
        print("-"*60)
        print(f"💰 Biaya Asuransi: Rp {prediction:,.2f}")
        print("="*60)
        
        return prediction
    
    def predict_batch(self, data):
        """
        Prediksi untuk banyak data
        
        Parameters:
        - data: DataFrame dengan kolom age, sex, bmi, children, smoker, region
        
        Returns:
        - DataFrame dengan tambahan kolom prediction
        """
        df = data.copy()
        
        # Encode
        df['sex_encoded'] = self.label_encoders['sex'].transform(df['sex'])
        df['smoker_encoded'] = self.label_encoders['smoker'].transform(df['smoker'])
        df['region_encoded'] = self.label_encoders['region'].transform(df['region'])
        
        # Buat array
        features = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']
        X = df[features].values
        
        # Standarisasi
        X_scaled = self.scaler.transform(X)
        
        # Prediksi
        predictions = self.model.predict(X_scaled)
        df['predicted_cost'] = predictions
        
        return df

# ==================== TEST PREDIKSI ====================
if __name__ == "__main__":
    # Buat object predictor
    predictor = InsurancePredictor()
    
    print("\n" + "="*60)
    print("CONTOH PREDIKSI")
    print("="*60)
    
    # Contoh 1: Perokok
    print("\n1. Pria, 30 tahun, BMI 30, perokok, 0 anak")
    pred1 = predictor.predict_with_details(30, 'male', 30.0, 0, 'yes', 'southeast')
    
    # Contoh 2: Bukan perokok
    print("\n2. Wanita, 30 tahun, BMI 25, bukan perokok, 1 anak")
    pred2 = predictor.predict_with_details(30, 'female', 25.0, 1, 'no', 'northwest')
    
    # Contoh 3: Usia tua, perokok
    print("\n3. Pria, 60 tahun, BMI 35, perokok, 2 anak")
    pred3 = predictor.predict_with_details(60, 'male', 35.0, 2, 'yes', 'northeast')
    
    # Contoh 4: Muda, sehat
    print("\n4. Wanita, 25 tahun, BMI 22, bukan perokok, 0 anak")
    pred4 = predictor.predict_with_details(25, 'female', 22.0, 0, 'no', 'southwest')
    
    # ==================== SAVE PREDICTION RESULTS ====================
    print("\n" + "="*60)
    print("SAVE PREDICTION RESULTS")
    print("="*60)
    
    # Buat DataFrame dari contoh prediksi
    example_data = pd.DataFrame({
        'age': [30, 30, 60, 25],
        'sex': ['male', 'female', 'male', 'female'],
        'bmi': [30.0, 25.0, 35.0, 22.0],
        'children': [0, 1, 2, 0],
        'smoker': ['yes', 'no', 'yes', 'no'],
        'region': ['southeast', 'northwest', 'northeast', 'southwest']
    })
    
    # Prediksi batch
    results = predictor.predict_batch(example_data)
    
    # Simpan ke CSV
    os.makedirs('results', exist_ok=True)
    results.to_csv('results/prediction_examples.csv', index=False)
    print("✅ Hasil prediksi disimpan ke 'results/prediction_examples.csv'")
    
    # Tampilkan hasil
    print("\nHasil Prediksi:")
    print(results[['age', 'sex', 'bmi', 'smoker', 'predicted_cost']].to_string(index=False))