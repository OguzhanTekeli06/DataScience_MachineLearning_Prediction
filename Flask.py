import json

import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, shapiro
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify

# Flask API uygulaması
app = Flask(__name__)

# JSON dosyasını yükleyin
file_path = 'C:\\Users\\ouzte\\Desktop\\veri\\data_science\\data\\audio_feature_with_mood.json'
with open(file_path, encoding='utf-8') as f:
    data = json.load(f)

# DataFrame'e dönüştürme
df = pd.json_normalize(data['audio_features'])

# Mood tahmini için veri hazırlığı
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']  # Özellikler
label = 'mood'  # Hedef değişken

# Eksik değerlerin doldurulması
df = df[features + [label]].dropna()

# Özellikler ve hedef değişkeni ayırma
X = df[features]
y = df[label]

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma ve eğitme
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test verisiyle modelin doğruluğunu kontrol etme
y_pred = model.predict(X_test)
print("Model Doğruluğu:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Flask API rotaları
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON formatındaki özellikleri al
        input_data = request.json
        input_df = pd.DataFrame([input_data])

        # Mood tahmini yap
        prediction = model.predict(input_df[features])
        return jsonify({'mood': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    app.run(debug=True)