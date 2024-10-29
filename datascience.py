import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# JSON verisini yükleyin
with open('C:\\Users\\ouzte\\Desktop\\veri\\data_science\\data\\audio_feature_with_mood.json', encoding='utf-8') as f:
    data = json.load(f)

# JSON verisindeki "audio_features" kısmını DataFrame'e çevirin
df = pd.json_normalize(data['audio_features'])

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
