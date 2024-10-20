import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Örnek veriyi yükleme
def load_data(file_path='data/audio_features.csv'):
    df = pd.read_csv(file_path)

    # Özellik sütunları
    X = df[['valence', 'energy', 'danceability', 'loudness', 'tempo', 'liveness']].values
    # Etiket sütunu
    y = df['mood'].values

    # Veriyi eğitim ve test seti olarak ayıralım
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
