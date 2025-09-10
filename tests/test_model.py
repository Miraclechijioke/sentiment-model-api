import pytest
import joblib
from train_and_evaluate import train_and_save_model

def test_model_training():
    # Train model
    vectorizer, model = train_and_save_model()
    
    # Check that the model is trained
    assert model is not None
    assert vectorizer is not None

    # Check simple prediction
    sample_text = ["I love AI"]
    X = vectorizer.transform(sample_text)
    pred = model.predict(X)
    assert pred[0] in [0, 1]


