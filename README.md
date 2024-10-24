# Audio Features Mood Classification
This project demonstrates how to build a machine learning model to predict mood based on audio features. Using audio data (valence, energy, danceability, etc.), a Random Forest Classifier is trained to classify moods, and feature importance is visualized.

## Project Structure
data_processing.py: Handles loading and preprocessing the JSON data.
model.py: Contains model training and evaluation functions.
visualization.py: Visualizes feature importance.

How to Use

Load Data
from data_processing import load_data_from_json
X_train, X_test, y_train, y_test = load_data_from_json('data/audio_feature_with_mood.json')


Train Model
from model import train_model
clf = train_model(X_train, y_train)


Evaluate Model
from model import evaluate_model
evaluate_model(clf, X_test, y_test)


Visualize Feature Importance
from visualization import plot_feature_importance
plot_feature_importance(clf)

## Dependencies
Install the required Python libraries:

pip install pandas numpy scikit-learn matplotlib

**File Descriptions**
data_processing.py: Loads and preprocesses audio feature data from a JSON file.
model.py: Contains functions for training and evaluating the Random Forest Classifier.
visualization.py: Visualizes the importance of features used by the classifier.

## Future Work
Improving model accuracy by experimenting with other classifiers.
Adding more features and tuning hyperparameters.

## License
This project is licensed under the MIT License.
