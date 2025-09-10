from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from train_and_evaluate import train_and_save_model
import logging
from logging.handlers import RotatingFileHandler
import joblib
import uvicorn
import os
import glob

# rotating file handler (max 5 MB per file, keep last 3 backups)
handler = RotatingFileHandler(
    "app.log", maxBytes=5*1024*1024, backupCount=3
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
               handler, # log to file
               logging.StreamHandler() # log to console
               ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

model_store = {"vectorizer": None, "model": None} # Start with None so the API can run even if model not loaded

def get_latest_model_file():
    """Return the latest sentiment_model_v*.pkl file based on timestamp in filename"""
    files = glob.glob("sentiment_model_v*.pkl")
    if not files:
        logger.warning(".pkl file doesn't exist in directory")
        return None
    latest_file = max(files, key=os.path.getctime)  # newest by creation time
    logger.info(".pkl file gotten from directory")
    return latest_file

# create load_model function
def load_model(auto_train=False):
    try:
        model_file = get_latest_model_file()
        if model_file:
            vectorizer, model = joblib.load(model_file)
            model_store["vectorizer"] = vectorizer
            model_store["model"] = model
            logger.info(f"Model loaded successfully from {model_file}")
        elif auto_train:
            logger.warning("No saved model found, training new model...")
            vectorizer, model = train_and_save_model()
            model_store["vectorizer"] = vectorizer
            model_store["model"] = model
        else:
            logger.warning("No model file found. API will run but predictions unavailable.")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)

# Load model at startup
load_model()

class InputText(BaseModel):
    text: str

'''
Note: BaseModel comes from Pydantic, which FastAPI uses for data validation.
This class says: “Any request to my API expecting an InputText must contain a JSON object
with one field: text, which must be a string.”
'''

# confirm that API is online
@app.get("/")
async def home():
    logger.info("Health check endpoint hit.")
    return {"message": "Sentiment Analysis API is running!"}


@app.post("/predict")
async def predict_sentiment(input: InputText):
    vectorizer = model_store.get("vectorizer")
    model = model_store.get("model")
    try:
        if not vectorizer or not model:
            logger.warning("Prediction requested but model not loaded.")
            raise HTTPException(status_code=503, detail="Model not loaded")
        X = vectorizer.transform([input.text])
        prediction = model.predict(X)[0] # class prediction
        probs = model.predict_proba(X)[0] # probability prediction
        confidence = probs[prediction]  # confidence for the class
        sentiment = "positive" if prediction == 1 else "negative"
        logger.info(f"Prediction made: sentiment={sentiment}, confidence={confidence:.4f}")
        return {"sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
def background_retrain():
    try:
        vectorizer, model = train_and_save_model()
        model_store["vectorizer"] = vectorizer
        model_store["model"] = model
        logger.info("Model retrained and reloaded successfully!")
    except Exception as e:
        logger.error(f"Retraining error: {e}", exc_info=True)


@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    try:
        """Start retraining in the background without blocking API requests."""
        background_tasks.add_task(background_retrain)
        logger.info("Retraining task started in background.")
        return {"message": "Retraining started in background. Check logs for progress."}
    except Exception as e:
        logger.error(f"Failed to start retraining: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
