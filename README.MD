# Sentiment Analysis API with FastAPI

A lightweight sentiment analysis API built with **FastAPI**. The API can predict the sentiment of text as **positive** or **negative**, return the confidence score, and retrain the model in the background. The model uses **Hugging Face IMDB dataset** with a **Logistic Regression + TF-IDF** pipeline.

---

## Features

- Train and save sentiment analysis model (`train_and_evaluate.py`)
- Predict sentiment via `/predict` endpoint
- Background retraining via `/retrain` endpoint
- Model loading and logging
- Optional Docker containerization
- CI/CD ready with GitHub Actions

---

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Deploy-an-ML-Model-with-FastAPI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
Open http://127.0.0.1:8080/docs to explore the API via Swagger UI.
```

## Project Structure

```bash
├── main.py                 # FastAPI app
├── train_and_evaluate.py   # Model training and saving
├── sentiment_model.pkl     # Saved model (after training)
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker setup
├── tests/                  # Unit tests
└── .github/workflows/      # CI/CD pipeline definitions
```

## Logging

The API uses Python logging to track:
- Model loading
- Prediction errors
- Retraining progress

## CI/CD

Supports GitHub Actions:
- Automatic testing on push
- Optional retraining workflow
- Deploys to Azure App Service or container registry

## Next Steps
- Deploy API to Azure App Service
- Add more robust unit tests
- Implement periodic retraining with new data
- Improve model with more advanced NLP techniques

## Author
Miracle Chijioke
- Data Scientist $ Machine Learning Engineer