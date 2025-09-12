from datasets import load_dataset

def test_imdb_dataset():
    dataset = load_dataset("imdb", split="train").shuffle(seed=42).select(range(1000))
    labels = set(dataset["label"])
    assert len(labels) >= 2, f"Expected at least 2 classes, got {labels}"
