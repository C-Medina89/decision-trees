import pandas as pd
from preprocess_data import convert_data, decode_data, clean_and_save_data

df = pd.read_csv("personality_dataset.csv")

# print(df.head())



encoders, encoded_df = convert_data(df)

encoded_list = encoded_df.iloc[0]
decoded_values = decode_data(encoded_list, encoders)

# print(encoded_df.head(), encoders)
print(decoded_values)


clean_and_save_data(df, "cleaned_dataset.csv")
