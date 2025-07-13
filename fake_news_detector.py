import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake['label'] = 0  # Fake = 0
true['label'] = 1  # True = 1

# Combine data
data = pd.concat([fake[['text', 'label']], true[['text', 'label']]])

# Shuffle dataset
data = data.sample(frac=1).reset_index(drop=True)

# Split into features and labels
X = data['text']
y = data['label']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Optional: Test with your own article
sample = input("\nEnter a news headline or article:\n")
sample_vec = vectorizer.transform([sample])
prediction = model.predict(sample_vec)
print("\nPrediction:", "Real News ✅" if prediction[0] == 1 else "Fake News ❌")
