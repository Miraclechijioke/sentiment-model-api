from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from train_and_evaluate import train_and_save_model
import joblib
import uvicorn
import os

app = FastAPI()

model_path = "sentiment_model.pkl"  # model path

model_store = {"vectorizer": None, "model": None} # Start with None so the API can run even if model not loaded


# create load_model function
def load_model(auto_train=False):
    try:
        if os.path.exists(model_path):
            vectorizer, model = joblib.load(model_path)
            model_store["vectorizer"] = vectorizer
            model_store["model"] = model
            print("✅ Model loaded successfully from disk!")
        elif auto_train:
            print("Model not found, starting training...")
            vectorizer, model = train_and_save_model()
            model_store["vectorizer"] = vectorizer
            model_store["model"] = model
        else:
            print("Model file not found. API will run, retrain needed.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

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
    return {"message": "Sentiment Analysis API is running!"}


@app.post("/predict")
async def predict_sentiment(input: InputText):
    vectorizer = model_store.get("vectorizer")
    model = model_store.get("model")
    try:
        if not vectorizer or not model:
            raise HTTPException(status_code=503, detail="Model not loaded")
        X = vectorizer.transform([input.text])
        prediction = model.predict(X)[0] # class prediction
        probs = model.predict_proba(X)[0] # probability prediction
        confidence = probs[prediction]  # confidence for the class
        sentiment = "positive" if prediction == 1 else "negative"
        return {"sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

def background_retrain():
    try:
        vectorizer, model = train_and_save_model()
        model_store["vectorizer"] = vectorizer
        model_store["model"] = model
        print("✅ Model retrained and reloaded successfully!")
    except Exception as e:
        print(f"❌ Retraining error: {e}")


@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    try:
        """Start retraining in the background without blocking API requests."""
        background_tasks.add_task(background_retrain)
        return {"message": "Retraining started in background. Check logs for progress."}
    except Exception as e:
        print(f"❌ Failed to start retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
