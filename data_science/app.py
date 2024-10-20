from data_processing import load_data
from model import train_model, evaluate_model
from visualization import plot_feature_importance

# 1. Veriyi yükleme ve hazırlama
X_train, X_test, y_train, y_test = load_data()

# 2. Modeli eğitme
clf = train_model(X_train, y_train)

# 3. Test seti üzerinde değerlendirme
evaluate_model(clf, X_test, y_test)

# 4. Özelliklerin önemini görselleştirme
plot_feature_importance(clf)
