# label_utils.py
from sklearn.preprocessing import LabelEncoder
import pickle

def label_encode_data(df):
    encoders = {}
    encoded_df = df.copy()
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        encoded_df[column] = le.fit_transform(df[column])
        encoders[column] = le
    return encoded_df, encoders

def save_encoders(encoders, file_path="models/label_encoders.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(encoders, f)

#Fungsi untuk memuat encoder
def load_encoders(file_path="label_encoders.pkl"):
    try:
        with open(file_path, "rb") as f:
            encoders = pickle.load(f)
        return encoders
    except FileNotFoundError:
        return None

#Fungsi untuk decoding data
def label_decode_data(encoded_df, encoders):
    decoded_df = encoded_df.copy()
    for column, le in encoders.items():
        decoded_df[column] = le.inverse_transform(encoded_df[column])
    return decoded_df