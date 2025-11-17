import pandas as pd
from src.split_data import split_data


def test_split_returns_four_outputs():
    X = pd.DataFrame({
    "age": [70, 30, 38, 21],
    "favourite_colour":[0, 1, 2, 1],
    "drives":[1, 0, 1, 1]
    }) 
    y = pd.Series([0,1,1,0])

    result = split_data(X,y, random_seed=1, test_size=0.4)

    assert len(result) == 4
    
    X_train, X_test, y_train, y_test = result
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None


def test_output_length():
    X = pd.DataFrame({
    "age": [70, 30, 38, 21],
    "favourite_colour":[0, 1, 2, 1],
    "drives":[1, 0, 1, 1]
    }) 
    y = pd.Series([0,1,1,0])

    X_train, X_test, y_train, y_test = split_data(X,y, random_seed=1, test_size=0.25)

    assert len(X_train) + len(X_test) == 4
    assert len(y_train) + len(y_test) == 4
    assert len(X_train) == 3
    assert len(y_train) == 3
    assert len(X_test) == 1
    assert len(y_test) == 1


def test_alignment_is_matching():

    X = pd.DataFrame({
    "age": [70, 30, 38, 21],
    "favourite_colour":[0, 1, 2, 1],
    "drives":[1, 0, 1, 1]
    }) 
    y = pd.Series([0,1,1,0])

    X_train, X_test, y_train, y_test = split_data(X,y, random_seed=1, test_size=0.25)



    train_recombined = X_train.copy()
    train_recombined["personality"] = y_train.values

    assert len(train_recombined) == len(X_train)

    test_recombined = X_test.copy()
    test_recombined["personality"] = y_test.values

    assert len(test_recombined) == len(X_test)
