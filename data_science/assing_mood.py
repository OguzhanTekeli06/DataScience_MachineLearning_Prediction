import json
import pandas as pd

def assign_mood(valence, energy, danceability):
    if valence > 0.7 and energy > 0.7 and danceability > 0.7:
        return "Neşeli ve Hareketli"
    elif valence > 0.6 and energy > 0.5 and danceability > 0.5:
        return "Hareketli"
    elif valence > 0.6 and energy < 0.5:
        return "Sakin ve Mutlu"
    elif valence < 0.4 and energy < 0.4:
        return "Melankolik"
    elif valence < 0.4 and energy > 0.4:
        return "Düşük Enerji"
    else:
        return "Karmaşık Ruh Hali"


# JSON dosyasını aç ve veriyi yükle
with open('data/audio_features.json', 'r') as f:
    json_data = json.load(f)

# Her şarkıya mood etiketi ekle
for track in json_data['audio_features']:
    valence = track['valence']
    energy = track['energy']
    danceability = track['danceability']
    track['mood'] = assign_mood(valence, energy, danceability)

# JSON dosyasını güncellenmiş haliyle kaydet
with open('data/audio_features_with_mood.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)
