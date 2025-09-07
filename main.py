from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from train_and_evaluate import train_and_save_model
import joblib
import uvicorn

app = FastAPI()

# Load model at startup
def load_model():
    vectorizer, model = joblib.load("sentiment_model.pkl")
    return vectorizer, model

vectorizer, model = load_model()


class InputText(BaseModel):
    text: str

'''
Note: BaseModel comes from Pydantic, which FastAPI uses for data validation.
This class says: “Any request to my API expecting an InputText must contain a JSON object
with one field: text, which must be a string.”
'''

# confirm that API is online
@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running!"}


@app.post("/predict")
async def predict_sentiment(input: InputText):
    X = vectorizer.transform([input.text])
    prediction = model.predict(X)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}

def background_retrain():
    global vectorizer, model
    vectorizer, model = train_and_save_model()
    print("✅ Model retrained and reloaded successfully!")

@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_retrain)
    return {"message": "Retraining started in background. Check logs for progress."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
