import joblib
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from datasets import load_dataset   # Hugging Face datasets


def train_and_save_model():  
    # Load IMDB dataset from Hugging Face
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # Extract texts and labels
    texts = dataset["train"]["text"]   # limit to 5k for speed
    labels = dataset["train"]["label"]

    # Train/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        list(texts), list(labels), test_size=0.3, random_state=42, stratify=list(labels)
    )

    # Train pipeline
    print("⚙️ Training model...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, train_labels)

    # Evaluate on test set
    X_test = vectorizer.transform(test_texts)
    preds = model.predict(X_test)

    print("✅ Accuracy:", accuracy_score(test_labels, preds))
    print("\n📊 Classification Report:\n", classification_report(test_labels, preds))
    

    version = int(time.time())  # Unix timestamp

    # Save pipeline
    joblib.dump((vectorizer, model), f"sentiment_model_v{version}.pkl")
    print(f"Model saved as sentiment_model_v{version}.pkl")

    # Return trained objects for use in main.py
    return vectorizer, model


# 👇 This runs only if script is executed directly
if __name__ == "__main__":
    print("Training model...")
    train_and_save_model()
    print("Training finished ✅")