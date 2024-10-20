import matplotlib.pyplot as plt
import numpy as np

# Özellik önemini görselleştirme fonksiyonu
def plot_feature_importance(clf):
    importances = clf.feature_importances_
    feature_names = ['valence', 'energy', 'danceability', 'loudness', 'tempo', 'liveness']
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Özelliklerin Önemi")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.show()
