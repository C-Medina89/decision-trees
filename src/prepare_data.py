import pandas as pd


def load_features_and_target(df):

    y = df["personality"]
    X = df.drop(columns=["personality"])

    return X,y