from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Model eğitme fonksiyonu
def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Model değerlendirme fonksiyonu
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
