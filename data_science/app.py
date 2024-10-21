from data_processing import load_data_from_json
from model import train_model, evaluate_model
from visualization import plot_feature_importance

# JSON dosyasını yükleyelim
X_train, X_test, y_train, y_test = load_data_from_json('data/audio_feature_with_mood.json')

# Modeli eğit ve sonuçları değerlendir
clf = train_model(X_train, y_train)
evaluate_model(clf, X_test, y_test)

# Özellik önemini görselleştir
plot_feature_importance(clf)
