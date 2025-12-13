import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# -----------------------------
# 1. Load Dataset
# -----------------------------
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0  # 0 = fake
real["label"] = 1  # 1 = real

df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

# -----------------------------
# 2. Clean Text Function
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z ]", " ", text)
    text = text.lower()
    return text

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# -----------------------------
# 3. Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Pipeline = TF-IDF + Logistic Regression
# -----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=50000)),
    ("clf", LogisticRegression(max_iter=300)),
])

# -----------------------------
# 5. Train
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate
# -----------------------------
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# -----------------------------
# 7. Save Model
# -----------------------------
joblib.dump(model, "models/fake_news_model.pkl")

print("Model trained and saved as models/fake_news_model.pkl")
