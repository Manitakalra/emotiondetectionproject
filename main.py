import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("dataset/emotions.csv")

X = data["text"]
y = data["emotion"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model
pickle.dump((model, vectorizer), open("saved_model.pkl", "wb"))

print("Model trained and saved!")
import pickle

# Load model
model, vectorizer = pickle.load(open("saved_model.pkl", "rb"))

print("😊 Emotion Detector Started (type 'exit' to stop)\n")

while True:
    text = input("Enter text: ")

    if text.lower() == "exit":
        break

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)

    print("Detected Emotion:", prediction[0])



