def split_data(X, y, random_seed, test_size=0.2):

    df = X.copy()
    df["personality"] = y

    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    test_len = int(len(df) * test_size)

    test_df = df.iloc[:test_len]
    train_df = df.iloc[test_len:]

    X_train = train_df.drop(columns=["personality"])
    y_train = train_df["personality"]

    X_test = test_df.drop(columns=["personality"])
    y_test = test_df["personality"]

    # print(X_train)
    return X_train, X_test , y_train, y_test




