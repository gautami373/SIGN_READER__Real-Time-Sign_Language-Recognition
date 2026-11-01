import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_DIR = "word_landmark_data_2hand"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "features.npy"))
y = np.load(os.path.join(DATA_DIR, "labels.npy"))

# Keep words with at least 10 samples
counts = Counter(y)
valid_words = [word for word, count in counts.items() if count >= 10]
mask = np.isin(y, valid_words)

X = X[mask]
y = y[mask]

print("Selected words:", len(valid_words))
print("Samples:", len(X))
print("Feature shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced_subsample",
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nTwo-hand model accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:\n")
print(classification_report(
    y_test, predictions, zero_division=0
))

joblib.dump(model, os.path.join(MODEL_DIR, "common_words_2hand_model.pkl"))
joblib.dump(valid_words, os.path.join(MODEL_DIR, "common_words_2hand_labels.pkl"))

print("\nModel saved successfully.")