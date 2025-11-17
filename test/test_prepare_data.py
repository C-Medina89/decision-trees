import pandas as pd
from src.prepare_data import load_features_and_target


def test_load_features_and_target():
    data ={
        "age": [70, 30, 38],
        "favourite_colour":[0, 1, 2],
        "drives":[1, 0, 1],
        "personality": [1, 0, 2]
    }

    df = pd.DataFrame(data)

    X, y = load_features_and_target(df)

    assert list(X.columns) == ["age", "favourite_colour", "drives"]
    assert y.name == "personality"
    assert list(y.values) == [1, 0, 2]
    assert "personality" not in X.columns