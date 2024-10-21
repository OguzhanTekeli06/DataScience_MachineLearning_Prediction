import json
import pandas as pd
from sklearn.model_selection import train_test_split

# JSON dosyasını yükleyen ve işleyen fonksiyon
def load_data_from_json(file_path='data/audio_features.json'):
    # JSON dosyasını aç ve veriyi yükle
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    # JSON verisini normalize edelim (veriyi tabloya çevirelim)
    df = pd.json_normalize(json_data['audio_features'])

    # Gerekli özellikleri seçelim (diğer sütunları çıkarıyoruz)
    X = df[['valence', 'energy', 'danceability', 'loudness', 'tempo', 'liveness', 'acousticness', 'speechiness', 'instrumentalness']].values

    # 'mood' etiketlerini ekleyelim veya yükleyelim
    y = df['mood']  # Eğer JSON'da mood sütunu varsa, doğrudan alınabilir

    # Veriyi eğitim ve test seti olarak ayıralım
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
