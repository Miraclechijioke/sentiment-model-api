from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from datasets import load_dataset   # Hugging Face datasets
import joblib
import numpy as np
import time
import logging
from logging.handlers import RotatingFileHandler

# rotating file handler (max 5 MB per file, keep last 3 backups)
handler = RotatingFileHandler(
    "app.log", maxBytes=5*1024*1024, backupCount=3
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        handler,  # log to file
        logging.StreamHandler()  # log to console
    ]
)

logger = logging.getLogger(__name__)


def train_and_save_model():  
    try:
        # Load IMDB dataset from Hugging Face
        logger.info("📥 Loading IMDB dataset...")
        dataset = load_dataset("imdb")

        # Extract texts and labels
        texts = dataset["train"]["text"]
        labels = dataset["train"]["label"]

        # Train/test split
        logger.info("Splitting dataset into train/test sets...")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            list(texts), list(labels), test_size=0.3, random_state=42, stratify=list(labels)
        )

        # Train pipeline
        logger.info("⚙️ Training model...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        X_train = vectorizer.fit_transform(train_texts)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, train_labels)

        # Evaluate on test set
        X_test = vectorizer.transform(test_texts)
        preds = model.predict(X_test)

        acc = accuracy_score(test_labels, preds)
        report = classification_report(test_labels, preds)

        logger.info(f"✅ Accuracy: {acc:.4f}")
        logger.info("📊 Classification Report:\n%s", report)

        # Save with timestamped filename
        version = int(time.time())  # Unix timestamp
        model_file = f"sentiment_model_v{version}.pkl"
        joblib.dump((vectorizer, model), model_file)

        logger.info(f"💾 Model saved as {model_file}")

        return vectorizer, model

    except Exception as e:
        logger.error("❌ Error occurred during training", exc_info=True)
        raise


# 👇 Runs only if script is executed directly
if __name__ == "__main__":
    logger.info("🚀 Starting training pipeline...")
    train_and_save_model()
    logger.info("🎉 Training finished successfully!")
