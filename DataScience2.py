import json
import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, shapiro

# JSON dosyasını yükleyin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

file_path = 'C:\\Users\\ouzte\\Desktop\\veri\\data_science\\data\\audio_feature_with_mood.json'
with open(file_path, encoding='utf-8') as f:
    data = json.load(f)

# DataFrame'e dönüştürme
df = pd.json_normalize(data['audio_features'])

# Ortalama değerler
print("Ortalama Değerler:")
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        print(f"{col}: {np.mean(df[col])}")

# Medyan değerler
print("\nMedyan Değerler:")
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        print(f"{col}: {df[col].median()}")

# Mod değerler
print("\nMod Değerler:")
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        try:
            print(f"{col}: {st.mode(df[col])}")
        except:
            print(f"{col}: Mod hesaplanamadı")

# Çarpıklık ve basıklık
print("\nÇarpıklık ve Basıklık:")
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        skewness = skew(df[col], bias=False)
        kurt = kurtosis(df[col], bias=False)
        print(f"{col} - Çarpıklık: {skewness}, Basıklık: {kurt}")

# Çeyreklik değerler
print("\nÇeyreklik Değerler:")
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        quantiles = np.percentile(df[col], [25, 50, 75])
        print(f"{col} - %25: {quantiles[0]}, Medyan (%50): {quantiles[1]}, %75: {quantiles[2]}")

# Histogram ve dağılım
numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
for col in numeric_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], kde=True, bins=20, color='skyblue')
    plt.title(f'{col} Dağılımı')
    plt.xlabel(col)
    plt.ylabel('Frekans')
    plt.show()

# Shapiro-Wilk testi
print("\nShapiro-Wilk Testi:")
for col in numeric_cols:
    stat, p_value = shapiro(df[col])
    print(f"{col} - Statistic: {stat}, p-value: {p_value}")

# Eksik veya hatalı değerlerin kontrolü
print("\nEksik veya Hatalı Değerler:")
print(df.isnull().sum())

# Verinin genel istatistik özeti
print("\nTanımlayıcı İstatistikler:")
print(df.describe())


# Gerekli sütunları seçin (mood dışında kalan özellikler)
features = ['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
            'speechiness', 'tempo', 'valence']
X = df[features]

# Mood sütununu hedef (target) olarak kullanın
y = df['mood']

# Veriyi eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli tanımlayın (örneğin, RandomForestClassifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test verileri üzerinde tahmin yapın
y_pred = model.predict(X_test)

# Modelin performansını değerlendirin
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk Oranı:", accuracy)
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Yeni şarkılar için mood tahmini yapabilirsiniz
new_song_features = X_test.iloc[0].values.reshape(1, -1)
predicted_mood = model.predict(new_song_features)
print("Tahmin Edilen Mood:", predicted_mood[0])

