import pandas as pd
from sklearn.preprocessing import LabelEncoder


def convert_data(df):

    processed_df = df.copy()
    encoded = {}


    for column in df.columns:
        if df[column].dtype == object:
            encoder = LabelEncoder()
            processed_df[column] = encoder.fit_transform(df[column])
            encoded[column] = encoder

    return encoded, processed_df


def decode_data(encoded_list, encoded):



    decoded_values = []
    col_names = list(encoded.keys())
    index = 0



    for column in col_names:
        # encoder =  LabelEncoder()
        encoder = encoded[column]
        # if column in encoded:
        decode_value = encoder.inverse_transform([encoded_list[column]])[0]
        decoded_values.append(decode_value)
        index += 1

    # decoded_values.extend(encoded_list[index:])


    return decoded_values



def clean_and_save_data(df, filepath):

    encoders, encoded_df = convert_data(df)
    encoded_df.to_csv(filepath, index=False)

    return encoders

    




    