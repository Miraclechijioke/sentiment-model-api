from datasets import load_dataset

def test_imdb_dataset():
    dataset = load_dataset("imdb", split="train[:1000]")  # small subset
    labels = set(dataset["label"])
    # Make sure there are at least 2 classes
    assert len(labels) >= 2
    assert all(label in [0,1] for label in labels)
