from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict():
    response = client.post("/predict", json={"text": "I love machine learning"})
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert response.json()["sentiment"] in ["positive", "negative"]

def test_retrain():
    response = client.post("/retrain")
    assert response.status_code == 200
    assert "message" in response.json()
