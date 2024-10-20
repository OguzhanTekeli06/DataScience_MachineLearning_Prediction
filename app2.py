import numpy as np

# Örnek JSON verisi
data = {
    "audio_features": [
        {"valence": 0.87, "energy": 0.874, "danceability": 0.702, "loudness": -9.863, "tempo": 95.036,
         "liveness": 0.129},
        {"valence": 0.557, "energy": 0.718, "danceability": 0.793, "loudness": -5.787, "tempo": 99.967,
         "liveness": 0.272},
        {"valence": 0.185, "energy": 0.537, "danceability": 0.619, "loudness": -9.155, "tempo": 75.994,
         "liveness": 0.105},
        {"valence": 0.654, "energy": 0.885, "danceability": 0.803, "loudness": -3.664, "tempo": 127.985,
         "liveness": 0.0503},
        {"valence": 0.198, "energy": 0.535, "danceability": 0.659, "loudness": -8.208, "tempo": 83.085,
         "liveness": 0.108},
        {"valence": 0.596, "energy": 0.94, "danceability": 0.816, "loudness": -6.079, "tempo": 128.0,
         "liveness": 0.431},
        {"valence": 0.158, "energy": 0.854, "danceability": 0.73, "loudness": -5.526, "tempo": 128.048,
         "liveness": 0.122},
        {"valence": 0.132, "energy": 0.509, "danceability": 0.57, "loudness": -12.303, "tempo": 149.961,
         "liveness": 0.0829},
        {"valence": 0.813, "energy": 0.88, "danceability": 0.69, "loudness": -3.525, "tempo": 159.933,
         "liveness": 0.126},
        {"valence": 0.542, "energy": 0.757, "danceability": 0.496, "loudness": -6.984, "tempo": 150.086,
         "liveness": 0.0731},
    ]
}

# Numpy dizisine dönüştürelim
valence = np.array([track['valence'] for track in data['audio_features']])
energy = np.array([track['energy'] for track in data['audio_features']])
danceability = np.array([track['danceability'] for track in data['audio_features']])
loudness = np.array([track['loudness'] for track in data['audio_features']])
tempo = np.array([track['tempo'] for track in data['audio_features']])
liveness = np.array([track['liveness'] for track in data['audio_features']])


# Şarkılar için skor tabanlı ruh hali tahmini fonksiyonu
def predict_mood(valence, energy, danceability, loudness, tempo, liveness):
    # Boş bir skor dizisi oluşturalım
    scores = np.zeros(len(valence))

    # Valence puanı ekle
    scores += np.where(valence > 0.7, 2, np.where(valence > 0.5, 1, 0))

    # Energy puanı ekle
    scores += np.where(energy > 0.7, 2, np.where(energy > 0.5, 1, 0))

    # Danceability puanı ekle
    scores += np.where(danceability > 0.7, 2, np.where(danceability > 0.5, 1, 0))

    # Tempo puanı ekle
    scores += np.where(tempo > 120, 2, np.where(tempo > 100, 1, 0))

    # Loudness (ses yüksekliği) puanı ekle
    scores += np.where(loudness > -5, 2, np.where(loudness > -10, 1, 0))

    # Liveness puanı ekle
    scores += np.where(liveness > 0.4, 1, 0)

    # Skor tabanlı ruh hali tahminleri
    moods = np.where(scores >= 8, 'Neşeli ve Hareketli',
                     np.where(scores >= 6, 'Hareketli',
                              np.where(scores >= 4, 'Orta Düzey Enerji', 'Düşük Enerji')))

    return moods


# Tahmin edilen ruh halleri
moods = predict_mood(valence, energy, danceability, loudness, tempo, liveness)

# Sonuçları yazdır
for i, mood in enumerate(moods):
    print(f"Şarkı {i + 1}: {mood}")
