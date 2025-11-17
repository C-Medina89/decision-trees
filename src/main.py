import pandas as pd
from src.preprocess_data import convert_data, decode_data, clean_and_save_data
from src.prepare_data import load_features_and_target

df = pd.read_csv("personality_dataset.csv")

# print(df.head())



encoders, encoded_df = convert_data(df)

encoded_list = encoded_df.iloc[0]
decoded_values = decode_data(encoded_list, encoders)

# print(encoded_df.head(), encoders)
# print(decoded_values)


clean_and_save_data(df, "cleaned_dataset.csv")

clean_df = pd.read_csv("cleaned_dataset.csv")
X, y = load_features_and_target(clean_df)

print(X.head())
print(y.head())