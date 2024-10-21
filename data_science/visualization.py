import matplotlib.pyplot as plt
import numpy as np


# Özellik önemini görselleştirme fonksiyonu
def plot_feature_importance(clf):
    # Özelliklerin önem değerlerini alalım
    importances = clf.feature_importances_

    # Tüm özellik isimlerini JSON'dan gelen veriyle uyumlu olacak şekilde tanımlayalım
    feature_names = ['valence', 'energy', 'danceability', 'loudness', 'tempo', 'liveness',
                     'acousticness', 'speechiness', 'instrumentalness']

    # Özellik önemlerini büyükten küçüğe sıralayalım
    indices = np.argsort(importances)[::-1]

    # Eğer özelliklerin sayısı, feature_names'ten daha büyükse, kontrol yapalım
    if len(importances) > len(feature_names):
        print("Warning: Feature importance length is greater than feature names length.")
        return

    # Grafik oluşturma
    plt.figure(figsize=(10, 6))
    plt.title("Özelliklerin Önemi")
    plt.bar(range(len(importances)), importances[indices], align="center")

    # Özellik isimlerini sıralanmış indekse göre yerleştirelim
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()
