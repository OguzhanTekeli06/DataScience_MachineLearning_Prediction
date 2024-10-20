import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Örnek veri (bu kısmı kendi verinle güncelleyebilirsin)
data = {
    "audio_features": [
        {"valence": 0.87, "energy": 0.874, "danceability": 0.702, "loudness": -9.863, "tempo": 95.036, "liveness": 0.129, "mood": "Neşeli ve Hareketli"},
        {"valence": 0.557, "energy": 0.718, "danceability": 0.793, "loudness": -5.787, "tempo": 99.967, "liveness": 0.272, "mood": "Hareketli"},
        {"valence": 0.185, "energy": 0.537, "danceability": 0.619, "loudness": -9.155, "tempo": 75.994, "liveness": 0.105, "mood": "Düşük Enerji"},
        # Diğer veriler...
    ]
}

# 1. Veriyi hazırlama
X = np.array([[track['valence'], track['energy'], track['danceability'], track['loudness'], track['tempo'], track['liveness']] for track in data['audio_features']])
y = np.array([track['mood'] for track in data['audio_features']])

# Veriyi eğitim ve test seti olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Modeli eğitme
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = clf.predict(X_test)

# 3. Sonuçları değerlendirme
print(classification_report(y_test, y_pred))

# 4. Özelliklerin önemini görselleştirme
importances = clf.feature_importances_
feature_names = ['valence', 'energy', 'danceability', 'loudness', 'tempo', 'liveness']
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Özelliklerin Önemi")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices])
plt.show()
