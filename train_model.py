import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("dataset/emotions.csv")

X = data["text"]
y = data["emotion"]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)

pickle.dump((model, vectorizer), open("saved_model.pkl", "wb"))

print("✅ Model trained and saved!")
